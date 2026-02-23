"""
panorama_sfm.pyベースの画像切り出し専用スクリプト
特徴抽出の前で処理を停止し、切り出し画像とマスクのみを生成
"""

import argparse
import json
import os
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import cv2
import numpy as np
import PIL.ExifTags
import PIL.Image
from scipy.spatial.transform import Rotation
from tqdm import tqdm

import pycolmap
from pycolmap import logging


@dataclass
class PanoRenderOptions:
    num_steps_yaw: int
    pitches_deg: Sequence[float]
    hfov_deg: float
    vfov_deg: float


PANO_RENDER_OPTIONS: dict[str, PanoRenderOptions] = {
    "overlapping": PanoRenderOptions(
        num_steps_yaw=4,
        pitches_deg=(-35.0, 0.0, 35.0),
        hfov_deg=90.0,
        vfov_deg=90.0,
    ),
    # Cubemap without top and bottom images.
    "non-overlapping": PanoRenderOptions(
        num_steps_yaw=4,
        pitches_deg=(0.0,),
        hfov_deg=90.0,
        vfov_deg=90.0,
    ),
}


def create_virtual_camera(
    pano_width: int,
    pano_height: int,
    hfov_deg: float,
    vfov_deg: float,
) -> pycolmap.Camera:
    """Create a virtual perspective camera."""
    image_width = int(pano_width * hfov_deg / 360)
    image_height = int(pano_height * vfov_deg / 180)
    focal = image_width / (2 * np.tan(np.deg2rad(hfov_deg) / 2))
    return pycolmap.Camera.create(
        0, "SIMPLE_PINHOLE", focal, image_width, image_height
    )


def get_virtual_camera_rays(camera: pycolmap.Camera) -> np.ndarray:
    size = (camera.width, camera.height)
    x, y = np.indices(size).astype(np.float32)
    xy = np.column_stack([x.ravel(), y.ravel()])
    # The center of the upper left most pixel has coordinate (0.5, 0.5)
    xy += 0.5
    xy_norm = camera.cam_from_img(xy)
    rays = np.concatenate([xy_norm, np.ones_like(xy_norm[:, :1])], -1)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    return rays


def spherical_img_from_cam(image_size, rays_in_cam: np.ndarray) -> np.ndarray:
    """Project rays into a 360 panorama (spherical) image."""
    if image_size[0] != image_size[1] * 2:
        raise ValueError("Only 360° panoramas are supported.")
    if rays_in_cam.ndim != 2 or rays_in_cam.shape[1] != 3:
        raise ValueError(f"{rays_in_cam.shape=} but expected (N,3).")
    r = rays_in_cam.T
    yaw = np.arctan2(r[0], r[2])
    pitch = -np.arctan2(r[1], np.linalg.norm(r[[0, 2]], axis=0))
    u = (1 + yaw / np.pi) / 2
    v = (1 - pitch * 2 / np.pi) / 2
    return np.stack([u, v], -1) * image_size


def get_virtual_rotations(
    num_steps_yaw: int, pitches_deg: Sequence[float]
) -> Sequence[np.ndarray]:
    """Get the relative rotations of the virtual cameras w.r.t. the panorama."""
    # Assuming that the panos are approximately upright.
    cams_from_pano_r = []
    yaws = np.linspace(0, 360, num_steps_yaw, endpoint=False)
    for pitch_deg in pitches_deg:
        yaw_offset = (360 / num_steps_yaw / 2) if pitch_deg > 0 else 0
        for yaw_deg in yaws + yaw_offset:
            cam_from_pano_r = Rotation.from_euler(
                "XY", [-pitch_deg, -yaw_deg], degrees=True
            ).as_matrix()
            cams_from_pano_r.append(cam_from_pano_r)
    return cams_from_pano_r


def create_pano_rig_config(
    cams_from_pano_rotation: Sequence[np.ndarray], ref_idx: int = 0
) -> pycolmap.RigConfig:
    """Create a RigConfig for the given virtual rotations."""
    rig_cameras = []
    for idx, cam_from_pano_rotation in enumerate(cams_from_pano_rotation):
        if idx == ref_idx:
            cam_from_rig = None
        else:
            cam_from_ref_rotation = (
                cam_from_pano_rotation @ cams_from_pano_rotation[ref_idx].T
            )
            cam_from_rig = pycolmap.Rigid3d(
                pycolmap.Rotation3d(cam_from_ref_rotation), np.zeros(3)
            )
        rig_cameras.append(
            pycolmap.RigConfigCamera(
                ref_sensor=idx == ref_idx,
                image_prefix=f"pano_camera{idx}/",
                cam_from_rig=cam_from_rig,
            )
        )
    return pycolmap.RigConfig(cameras=rig_cameras)


class PanoProcessor:
    def __init__(
        self,
        pano_image_dir: Path,
        output_image_dir: Path,
        mask_dir: Path,
        render_options: PanoRenderOptions,
    ):
        self.render_options = render_options
        self.pano_image_dir = pano_image_dir
        self.output_image_dir = output_image_dir
        self.mask_dir = mask_dir

        self.cams_from_pano_rotation = get_virtual_rotations(
            num_steps_yaw=render_options.num_steps_yaw,
            pitches_deg=render_options.pitches_deg,
        )
        self.rig_config = create_pano_rig_config(self.cams_from_pano_rotation)

        # We assign each pano pixel to the virtual camera
        # with the closest camera center.
        self.cam_centers_in_pano = np.einsum(
            "nij,i->nj", self.cams_from_pano_rotation, [0, 0, 1]
        )

        self._lock = Lock()

        # These are initialized on the first pano image
        # to avoid recomputing the rays for each pano image.
        self._camera = None
        self._pano_size = None
        self._rays_in_cam = None

    def process(self, pano_name: str):
        pano_path = self.pano_image_dir / pano_name
        try:
            pano_image = PIL.Image.open(pano_path)
        except PIL.Image.UnidentifiedImageError:
            logging.info(f"Skipping file {pano_path} as it cannot be read.")
            return

        pano_exif = pano_image.getexif()
        pano_image = np.asarray(pano_image)
        gpsonly_exif = PIL.Image.Exif()
        gpsonly_exif[PIL.ExifTags.IFD.GPSInfo] = pano_exif.get_ifd(
            PIL.ExifTags.IFD.GPSInfo
        )

        pano_height, pano_width, *_ = pano_image.shape
        if pano_width != pano_height * 2:
            raise ValueError("Only 360° panoramas are supported.")

        with self._lock:
            if self._camera is None:  # First image, precompute rays once.
                self._camera = create_virtual_camera(
                    pano_width=pano_width,
                    pano_height=pano_height,
                    hfov_deg=self.render_options.hfov_deg,
                    vfov_deg=self.render_options.vfov_deg,
                )
                for rig_camera in self.rig_config.cameras:
                    rig_camera.camera = self._camera
                self._pano_size = (pano_width, pano_height)
                self._rays_in_cam = get_virtual_camera_rays(self._camera)
            else:  # Later images, verify consistent panoramas.
                if (pano_width, pano_height) != self._pano_size:
                    raise ValueError(
                        "Panoramas of different sizes are not supported."
                    )

        for cam_idx, cam_from_pano_r in enumerate(self.cams_from_pano_rotation):
            rays_in_pano = self._rays_in_cam @ cam_from_pano_r
            xy_in_pano = spherical_img_from_cam(self._pano_size, rays_in_pano)
            xy_in_pano = xy_in_pano.reshape(
                self._camera.width, self._camera.height, 2
            ).astype(np.float32)
            xy_in_pano -= 0.5  # COLMAP to OpenCV pixel origin.
            image = cv2.remap(
                pano_image,
                *np.moveaxis(xy_in_pano, [0, 1, 2], [2, 1, 0]),
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_WRAP,
            )
            # We define a mask such that each pixel of the panorama has its
            # features extracted only in a single virtual camera.
            closest_camera = np.argmax(
                rays_in_pano @ self.cam_centers_in_pano.T, -1
            )
            mask = (
                ((closest_camera == cam_idx) * 255)
                .astype(np.uint8)
                .reshape(self._camera.width, self._camera.height)
                .transpose()
            )

            image_name = (
                self.rig_config.cameras[cam_idx].image_prefix + pano_name
            )
            mask_name = f"{image_name}.png"

            image_path = self.output_image_dir / image_name
            image_path.parent.mkdir(exist_ok=True, parents=True)
            PIL.Image.fromarray(image).save(image_path, exif=gpsonly_exif)

            mask_path = self.mask_dir / mask_name
            mask_path.parent.mkdir(exist_ok=True, parents=True)
            if not pycolmap.Bitmap.from_array(mask).write(mask_path):
                raise RuntimeError(f"Cannot write {mask_path}")


def render_perspective_images(
    pano_image_names: Sequence[str],
    pano_image_dir: Path,
    output_image_dir: Path,
    mask_dir: Path,
    render_options: PanoRenderOptions,
) -> pycolmap.RigConfig:
    processor = PanoProcessor(
        pano_image_dir, output_image_dir, mask_dir, render_options
    )

    num_panos = len(pano_image_names)
    max_workers = min(32, (os.cpu_count() or 2) - 1)

    with tqdm(total=num_panos) as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as thread_pool:
            futures = [
                thread_pool.submit(processor.process, pano_name)
                for pano_name in pano_image_names
            ]
            for future in as_completed(futures):
                future.result()
                pbar.update(1)

    return processor.rig_config


def save_rig_config_to_json(rig_config: pycolmap.RigConfig, output_path: Path):
    """RigConfigをJSON形式で保存（後で使えるように）"""
    rig_dict = {
        "cameras": []
    }
    
    for idx, cam in enumerate(rig_config.cameras):
        cam_dict = {
            "image_prefix": cam.image_prefix,
            "ref_sensor": cam.ref_sensor,
        }
        
        if cam.camera is not None:
            # pycolmapのバージョンによって属性名が異なる対応
            try:
                model_name = cam.camera.model_name
            except AttributeError:
                try:
                    model_name = cam.camera.model.name
                except AttributeError:
                    # model_idから名前を取得
                    model_name = cam.camera.ModelName()
            
            cam_dict["camera"] = {
                "model": model_name,
                "width": cam.camera.width,
                "height": cam.camera.height,
                "params": cam.camera.params.tolist(),
            }
        
        if cam.cam_from_rig is not None:
            cam_dict["cam_from_rig_rotation"] = cam.cam_from_rig.rotation.quat.tolist()
            cam_dict["cam_from_rig_translation"] = cam.cam_from_rig.translation.tolist()
        
        rig_dict["cameras"].append(cam_dict)
    
    with open(output_path, 'w') as f:
        json.dump(rig_dict, f, indent=2)
    
    logging.info(f"Saved rig config to {output_path}")


def run(args):
    pycolmap.set_random_seed(0)

    # Define the paths.
    image_dir = args.output_path / "images"
    mask_dir = args.output_path / "masks"
    image_dir.mkdir(exist_ok=True, parents=True)
    mask_dir.mkdir(exist_ok=True, parents=True)

    database_path = args.output_path / "database.db"
    
    # Search for input images.
    pano_image_dir = args.input_image_path
    pano_image_names = sorted(
        p.relative_to(pano_image_dir).as_posix()
        for p in pano_image_dir.rglob("*")
        if not p.is_dir()
    )
    logging.info(f"Found {len(pano_image_names)} images in {pano_image_dir}.")

    # ========================================
    # ステップ1: 画像切り出しとマスク生成
    # ========================================
    
    logging.info("=" * 60)
    logging.info("ステップ1: 画像切り出しとマスク生成")
    logging.info("=" * 60)
    
    rig_config = render_perspective_images(
        pano_image_names,
        pano_image_dir,
        image_dir,
        mask_dir,
        PANO_RENDER_OPTIONS[args.pano_render_type],
    )

    # RigConfigをJSON形式で保存
    rig_config_path = args.output_path / "rig_config.json"
    save_rig_config_to_json(rig_config, rig_config_path)

    logging.info("✓ 画像切り出しとマスク生成が完了")
    
    # ========================================
    # ステップ2: 特徴抽出
    # ========================================
    
    rig_config_applied = False  # Rig Config適用状況を追跡
    
    if not args.skip_feature_extraction:
        logging.info("")
        logging.info("=" * 60)
        logging.info("ステップ2: 特徴抽出")
        logging.info("=" * 60)
        
        # データベースが既に存在する場合は削除
        if database_path.exists():
            logging.info(f"既存のデータベースを削除: {database_path}")
            database_path.unlink()
        
        # 特徴抽出（動作するバージョンのAPIを使用）
        pycolmap.extract_features(
            database_path,
            image_dir,
            reader_options={"mask_path": mask_dir},
            camera_mode=pycolmap.CameraMode.PER_FOLDER,
        )
        
        logging.info("✓ 特徴抽出完了")
        
        # ========================================
        # ステップ3: Rig Config適用
        # ========================================
        
        if not args.skip_rig_config:
            logging.info("")
            logging.info("=" * 60)
            logging.info("ステップ3: Rig Config適用")
            logging.info("=" * 60)
            
            # 動作するバージョンのAPIを使用
            try:
                with pycolmap.Database.open(database_path) as db:
                    pycolmap.apply_rig_config([rig_config], db)
                logging.info("✓ Rig設定を適用しました")
                rig_config_applied = True
            except Exception as e:
                logging.error(f"Rig Config適用エラー: {e}")
                logging.warning("Rig Configの適用に失敗しましたが、処理を続行します")
                logging.warning("マスクによる重複防止は有効です")
    
    # rig_config_applied変数を設定（skip_rig_configの場合）
    if args.skip_rig_config:
        rig_config_applied = False
    
    # ========================================
    # 完了メッセージ
    # ========================================
    
    logging.info("")
    logging.info("=" * 60)
    logging.info("処理完了")
    logging.info("=" * 60)
    logging.info(f"切り出し画像: {image_dir}")
    logging.info(f"マスク: {mask_dir}")
    logging.info(f"Rig設定: {rig_config_path}")
    
    if not args.skip_feature_extraction:
        logging.info(f"データベース: {database_path}")
        if rig_config_applied:
            logging.info("Rig Config: 適用済み")
        else:
            logging.info("Rig Config: 未適用（Exhaustive Matcher推奨）")
        logging.info("")
        logging.info("次のステップ:")
        logging.info("1. DSLR写真を追加する場合:")
        logging.info(f"   cp dslr_photos/* {image_dir}/")
        logging.info(f"   colmap feature_extractor --database_path {database_path} --image_path {image_dir}")
        logging.info("")
        if rig_config_applied:
            logging.info("2. マッチング（Sequential推奨）:")
            logging.info(f"   colmap sequential_matcher --database_path {database_path}")
        else:
            logging.info("2. マッチング（Exhaustive必須 - Rig Config未適用）:")
            logging.info(f"   colmap exhaustive_matcher --database_path {database_path}")
        logging.info("")
        logging.info("3. マッピング:")
        logging.info(f"   colmap mapper --database_path {database_path} --image_path {image_dir} --output_path sparse/")
    else:
        logging.info("")
        logging.info("次のステップ:")
        logging.info("1. DSLR写真を追加する場合:")
        logging.info(f"   cp dslr_photos/* {image_dir}/")
        logging.info("")
        logging.info("2. COLMAPで特徴抽出:")
        logging.info(f"   colmap feature_extractor \\")
        logging.info(f"     --database_path {database_path} \\")
        logging.info(f"     --image_path {image_dir} \\")
        logging.info(f"     --ImageReader.mask_path {mask_dir} \\")
        logging.info(f"     --ImageReader.single_camera_per_folder 1")
        logging.info("")
        logging.info("3. マッチングとマッピング:")
        logging.info("   colmap sequential_matcher ...")
        logging.info("   colmap mapper ...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="panorama_sfm.pyベースの画像切り出し・特徴抽出・Rig Config適用スクリプト"
    )
    parser.add_argument("--input_image_path", type=Path, required=True,
                        help="360度画像の入力ディレクトリ")
    parser.add_argument("--output_path", type=Path, required=True,
                        help="出力ディレクトリ")
    parser.add_argument(
        "--pano_render_type",
        default="overlapping",
        choices=list(PANO_RENDER_OPTIONS.keys()),
        help="切り出しタイプ（デフォルト: overlapping = 4方向×3ピッチ=12視点）"
    )
    parser.add_argument(
        "--skip_feature_extraction",
        action="store_true",
        help="特徴抽出をスキップ（画像切り出しとマスク生成のみ）"
    )
    parser.add_argument(
        "--skip_rig_config",
        action="store_true",
        help="Rig Config適用をスキップ"
    )
    run(parser.parse_args())
