"""COLMAP画像リネームスクリプト

pano0/image.jpg のように複数のpanoディレクトリで同名ファイルが存在する問題を解決します。
pano0/image.jpg → pano0/pano0_image00001.jpg のようにリネームします。

更新対象:
  - imagesフォルダ内の実ファイル
  - database.db の images.name
  - sparse/*/images.txt の NAME フィールド (存在する場合)
  - masksフォルダ内の実ファイル (--update-masks 指定時)

使用例:
  # 確認のみ (ファイル変更なし)
  python rename_colmap_images.py --output_path /path/to/output --dry-run

  # 実際にリネーム
  python rename_colmap_images.py --output_path /path/to/output

  # マスクも一緒にリネーム
  python rename_colmap_images.py --output_path /path/to/output --update-masks
"""

import argparse
import shutil
import sqlite3
from pathlib import Path


def build_rename_map(image_names: list[str]) -> dict[str, str]:
    """画像名のリネームマップを構築する。

    Args:
        image_names: "pano0/image.jpg" 形式の相対パスリスト

    Returns:
        {old_name: new_name} の辞書
        例: {"pano0/image.jpg": "pano0/pano0_image00001.jpg"}
    """
    # 親ディレクトリごとにグループ化
    groups: dict[str, list[str]] = {}
    for name in image_names:
        parent = Path(name).parent.as_posix()
        groups.setdefault(parent, []).append(name)

    rename_map: dict[str, str] = {}
    for parent_dir, names in groups.items():
        dir_prefix = Path(parent_dir).name  # "pano0" など
        for idx, name in enumerate(sorted(names), start=1):
            p = Path(name)
            new_name = str(p.parent / f"{dir_prefix}_{p.stem}{idx:05d}{p.suffix}")
            rename_map[name] = new_name

    return rename_map


def collect_image_names_from_db(db_path: Path) -> list[str]:
    """DBのimagesテーブルから画像名一覧を取得する。"""
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute("SELECT name FROM images ORDER BY name")
        return [row[0] for row in cur.fetchall()]
    finally:
        conn.close()


def collect_image_names_from_dir(images_dir: Path) -> list[str]:
    """imagesフォルダから画像名一覧を収集する。"""
    return sorted(
        p.relative_to(images_dir).as_posix()
        for p in images_dir.rglob("*")
        if p.is_file()
    )


def rename_files(base_dir: Path, rename_map: dict[str, str], label: str) -> int:
    """フォルダ内のファイルをリネームする。

    Returns:
        リネームしたファイル数
    """
    count = 0
    for old_rel, new_rel in sorted(rename_map.items()):
        old_path = base_dir / old_rel
        new_path = base_dir / new_rel
        if not old_path.exists():
            print(f"  [SKIP] {old_rel} not found in {label}")
            continue
        new_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(old_path), str(new_path))
        print(f"  {old_rel} → {new_rel}")
        count += 1
    return count


def update_database(db_path: Path, rename_map: dict[str, str]) -> int:
    """database.db の images.name を更新する。

    Returns:
        更新した行数
    """
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        count = 0
        for old_name, new_name in rename_map.items():
            cur.execute(
                "UPDATE images SET name = ? WHERE name = ?",
                (new_name, old_name),
            )
            count += cur.rowcount
        conn.commit()
        return count
    finally:
        conn.close()


def update_images_txt(images_txt_path: Path, rename_map: dict[str, str]) -> int:
    """sparse/*/images.txt の NAME フィールドを更新する。

    images.txt の形式:
      # コメント行
      IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME  ← 奇数行
      X Y POINT3D_ID X Y POINT3D_ID ...             ← 偶数行 (2D points)

    Returns:
        更新した画像エントリ数
    """
    lines = images_txt_path.read_text().splitlines(keepends=True)
    new_lines = []
    count = 0
    # コメントを除いた非空行で、交互に画像行/2Dポイント行となる
    data_line_idx = 0

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#") or stripped == "":
            new_lines.append(line)
            continue

        # 偶数番目のデータ行が画像情報行 (0-indexed: 0, 2, 4, ...)
        if data_line_idx % 2 == 0:
            # IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
            # NAME はスペースを含まない前提
            parts = stripped.split(" ", 9)
            if len(parts) == 10:
                old_name = parts[9].rstrip("\n")
                if old_name in rename_map:
                    parts[9] = rename_map[old_name]
                    count += 1
                new_lines.append(" ".join(parts) + "\n")
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)

        data_line_idx += 1

    images_txt_path.write_text("".join(new_lines))
    return count


def run(args: argparse.Namespace) -> None:
    output_path: Path = args.output_path
    images_dir = output_path / "images"
    masks_dir = output_path / "masks"
    db_path = output_path / "database.db"
    sparse_path = output_path / "sparse"

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    # 画像名一覧を取得 (DB優先、なければフォルダから)
    db_names = collect_image_names_from_db(db_path)
    dir_names = collect_image_names_from_dir(images_dir)

    if db_names:
        print(f"Found {len(db_names)} images in database.")
        image_names = db_names
    else:
        print(f"Database is empty. Using {len(dir_names)} images from folder.")
        image_names = dir_names

    if not image_names:
        print("No images found. Nothing to do.")
        return

    # リネームマップを構築
    rename_map = build_rename_map(image_names)
    no_change = {k: v for k, v in rename_map.items() if k == v}
    actual_renames = {k: v for k, v in rename_map.items() if k != v}

    if no_change:
        print(f"  {len(no_change)} images already have unique names (skipped).")
    print(f"  {len(actual_renames)} images will be renamed.")

    if not actual_renames:
        print("Nothing to rename.")
        return

    if args.dry_run:
        print("\n[DRY RUN] The following renames would be performed:")
        for old, new in sorted(actual_renames.items()):
            print(f"  {old} → {new}")
        return

    # 1. 実ファイルをリネーム
    print("\nRenaming image files...")
    n = rename_files(images_dir, actual_renames, "images")
    print(f"  → {n} files renamed.")

    # 2. マスクファイルをリネーム (オプション)
    if args.update_masks and masks_dir.exists():
        # masks は "画像相対パス.png" の形式
        mask_rename_map = {
            old + ".png": new + ".png"
            for old, new in actual_renames.items()
        }
        print("\nRenaming mask files...")
        n = rename_files(masks_dir, mask_rename_map, "masks")
        print(f"  → {n} files renamed.")

    # 3. DBを更新
    print("\nUpdating database...")
    n = update_database(db_path, actual_renames)
    print(f"  → {n} rows updated.")

    # 4. images.txt を更新 (sparse/ 以下を再帰的に検索)
    if sparse_path.exists():
        txt_files = list(sparse_path.rglob("images.txt"))
        if txt_files:
            for images_txt in txt_files:
                print(f"\nUpdating {images_txt.relative_to(output_path)}...")
                n = update_images_txt(images_txt, actual_renames)
                print(f"  → {n} entries updated.")
        else:
            print("\nNo images.txt found in sparse/.")

    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Rename COLMAP images to include pano directory prefix and sequential number.\n"
            "e.g. pano0/image.jpg → pano0/pano0_image00001.jpg"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        required=True,
        help="Path to COLMAP output directory (must contain images/ and database.db)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be renamed without actually making changes",
    )
    parser.add_argument(
        "--update-masks",
        action="store_true",
        help="Also rename files in masks/ directory (assumes mask = image_path + '.png')",
    )
    run(parser.parse_args())
