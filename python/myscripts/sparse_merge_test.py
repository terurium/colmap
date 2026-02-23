import pycolmap
import numpy as np
import os
import shutil
import sys

def merge_sparse_for_3dgs(sparse1_path, sparse2_path, output_path, 
                          validate_gui=True, strict_mode=True):
    """
    3DGS用に2つのsparse reconstructionを精度重視でマージ
    """
    
    print("=== 3DGS向けSparse Reconstructionマージ ===")
    
    # 1. Reconstructionの読み込み
    print("\n[1/6] Reconstructionを読み込み中...")
    print(f"  Reconstruction 1: {sparse1_path}")
    print(f"  Reconstruction 2: {sparse2_path}")
    
    rec1 = pycolmap.Reconstruction(sparse1_path)
    rec2 = pycolmap.Reconstruction(sparse2_path)
    
    print(f"  Reconstruction 1: {len(rec1.images)}枚, {len(rec1.points3D)}点")
    print(f"  Reconstruction 2: {len(rec2.images)}枚, {len(rec2.points3D)}点")
    
    # 初期品質チェック
    error1 = rec1.compute_mean_reprojection_error()
    error2 = rec2.compute_mean_reprojection_error()
    print(f"  再投影誤差: Rec1={error1:.3f}px, Rec2={error2:.3f}px")
    
    if strict_mode and (error1 > 3.0 or error2 > 3.0):
        print("  ⚠️ 警告: 元のreconstructionの誤差が高いです")
        response = input("  続行しますか？ (y/n): ")
        if response.lower() != 'y':
            return None
    
    # 2. 共通情報の確認
    print("\n[2/6] 共通情報を確認中...")
    images1 = set(rec1.images.keys())
    images2 = set(rec2.images.keys())
    common_images = images1 & images2
    
    print(f"  共通画像ID数: {len(common_images)}")
    print(f"  Rec1のみの画像: {len(images1 - images2)}")
    print(f"  Rec2のみの画像: {len(images2 - images1)}")
    
    if len(common_images) == 0:
        print("\n  ⚠️ 共通画像IDが存在しません")
        print("  これは別々のデータベースで処理されたことを意味します")
        print("\n  このケースでは、以下のアプローチが必要です:")
        print("    1. データベース統合アプローチ（推奨）")
        print("    2. GPS/EXIF情報を使った初期アライメント")
        print("    3. 手動アライメント")
        print("\n  現在のマージは中止します。")
        return None
    
    # 3. GUIでの視覚的確認（推奨）
    if validate_gui:
        print("\n[3/6] COLMAP GUIで両方を確認してください:")
        print(f"  colmap gui --import_path {sparse1_path}")
        print(f"  colmap gui --import_path {sparse2_path}")
        print("  確認事項:")
        print("    - カメラ軌跡が自然か")
        print("    - tanglingがないか")
        print("    - 重複する領域があるか")
        input("  確認完了後、Enterを押してください...")
    
    # 4. アライメント（共通画像がある場合）
    print(f"\n[4/6] Reconstructionをアライメント中...")
    print(f"  共通画像を使用: {len(common_images)}枚")
    
    try:
        # 共通画像の再投影を使ったアライメント
        sim3 = pycolmap.align_reconstructions_via_reprojections(
            rec1,  # src_reconstruction
            rec2,  # tgt_reconstruction
            max_reproj_error=8.0  # パノラマの場合は緩めに
        )
        
        if sim3 is None:
            raise RuntimeError("アライメントが収束しませんでした")
        
        print(f"  アライメント成功:")
        print(f"    回転: {sim3.rotation}")
        print(f"    並進: {sim3.translation}")
        print(f"    スケール: {sim3.scale}")
        
        # 変換を適用
        rec2.transform(sim3)
        
    except (RuntimeError, TypeError, IndexError) as e:
        print(f"  ❌ 自動アライメント失敗: {e}")
        print("\n  解決策:")
        print("    データベース統合アプローチを使用してください")
        return None
    
    # 5. マージ
    print("\n[5/6] Reconstructionをマージ中...")
    
    merged = pycolmap.merge_reconstructions(
        rec1, 
        rec2,
        max_reproj_error=2.0 if strict_mode else 4.0
    )
    
    print(f"  マージ後: {len(merged.images)}枚, {len(merged.points3D)}点")
    
    # 6. Global Bundle Adjustment（最重要！）
    print("\n[6/6] Global Bundle Adjustmentを実行中...")
    
    ba_options = pycolmap.BundleAdjustmentOptions()
    
    # 3DGS用の厳密な設定
    ba_options.refine_focal_length = True
    ba_options.refine_principal_point = True
    ba_options.refine_extra_params = True
    ba_options.max_num_iterations = 100 if strict_mode else 50
    ba_options.function_tolerance = 1e-6
    ba_options.gradient_tolerance = 1e-10
    
    # BAを実行
    summary = merged.bundle_adjustment(ba_options)
    
    print(f"  BA完了:")
    print(f"    初期コスト: {summary.initial_cost:.6f}")
    print(f"    最終コスト: {summary.final_cost:.6f}")
    print(f"    反復回数: {summary.num_iterations}")
    
    # 7. 品質検証
    print("\n[7/7] 品質検証中...")
    validate_reconstruction_for_3dgs(merged, strict_mode)
    
    # 保存
    os.makedirs(output_path, exist_ok=True)
    merged.write(output_path)
    print(f"\n✓ マージされたreconstructionを保存: {output_path}")
    
    return output_path


def validate_reconstruction_for_3dgs(rec, strict_mode=True):
    """
    3DGS用のreconstruction品質を検証
    """
    print("\n  === 3DGS品質チェック ===")
    
    # 1. 基本統計
    num_images = len(rec.images)
    num_points = len(rec.points3D)
    mean_obs = rec.compute_mean_observations()
    mean_error = rec.compute_mean_reprojection_error()
    
    print(f"    画像数: {num_images}")
    print(f"    3D点数: {num_points}")
    print(f"    平均観測数: {mean_obs:.2f}")
    print(f"    平均再投影誤差: {mean_error:.3f}px")
    
    # 2. 再投影誤差の分布
    errors = [point.error for point in rec.points3D.values()]
    errors = np.array(errors)
    
    print(f"    誤差中央値: {np.median(errors):.3f}px")
    print(f"    誤差90%点: {np.percentile(errors, 90):.3f}px")
    print(f"    誤差95%点: {np.percentile(errors, 95):.3f}px")
    print(f"    誤差最大: {np.max(errors):.3f}px")
    
    # 3. カメラの分布
    positions = np.array([img.projection_center() for img in rec.images.values()])
    
    print(f"    カメラ範囲:")
    print(f"      X: [{positions[:,0].min():.2f}, {positions[:,0].max():.2f}]")
    print(f"      Y: [{positions[:,1].min():.2f}, {positions[:,1].max():.2f}]")
    print(f"      Z: [{positions[:,2].min():.2f}, {positions[:,2].max():.2f}]")
    
    # 4. 警告とエラー
    issues = []
    
    if mean_error > 2.0:
        issues.append(f"平均再投影誤差が高い ({mean_error:.3f}px > 2.0px)")
    
    if np.percentile(errors, 95) > 5.0:
        issues.append(f"95%点の誤差が高い ({np.percentile(errors, 95):.3f}px > 5.0px)")
    
    if mean_obs < 3.0:
        issues.append(f"平均観測数が少ない ({mean_obs:.2f} < 3.0)")
    
    if num_points < 1000:
        issues.append(f"3D点数が少ない ({num_points} < 1000)")
    
    if issues:
        print("\n  ⚠️ 検出された問題:")
        for issue in issues:
            print(f"    - {issue}")
        
        if strict_mode:
            print("\n  推奨事項:")
            print("    - COLMAP GUIで視覚的に確認")
            print("    - パラメータ調整を検討")
    else:
        print("\n  ✓ 品質チェック合格！")


# メイン処理
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("使用法: python sparse_merge_test.py <sparse1_path> <sparse2_path> [output_path]")
        print("例: python sparse_merge_test.py video1/sparse/0 video2/sparse/0 merged/sparse/0")
        sys.exit(1)
    
    sparse1 = sys.argv[1]
    sparse2 = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) > 3 else "merged/sparse/0"
    
    # マージ実行
    result = merge_sparse_for_3dgs(
        sparse1, 
        sparse2, 
        output,
        validate_gui=False,  # コマンドライン実行時はスキップ
        strict_mode=True
    )
    
    if result is None:
        print("\n❌ マージ失敗")
        print("\n=== データベース統合アプローチを推奨 ===")
        print("別々のデータベースで処理された2つのreconstructionは")
        print("直接マージできません。以下の方法を試してください:")
        print("\n1. 最初から統合処理（最も確実）:")
        print("   - 両方の画像を1つのディレクトリに配置")
        print("   - 単一のデータベースでfeature extraction & matching")
        print("   - 1回のmappingで処理")
        print("\n2. 個別に処理してから点群レベルでマージ:")
        print("   - 各reconstructionをdense化")
        print("   - 点群をOpen3D等でマージ")
        sys.exit(1)
    else:
        print(f"\n✓ 成功: {result}")
        print("\n次のステップ:")
        print(f"  colmap gui --import_path {result}")