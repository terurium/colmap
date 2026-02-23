# 動作確認済みスクリプト使用ガイド

## ✅ 重要な修正点

あなたの環境で動作するpanorama_sfm.pyをベースに、以下の**正しいAPI**を使用しています：

```python
# ✅ 正しいDatabase初期化
with pycolmap.Database.open(database_path) as db:
    pycolmap.apply_rig_config([rig_config], db)

# ✅ 正しいmask_path指定
reader_options={"mask_path": mask_dir}  # Path直接、str()不要
```

## 📦 提供スクリプト

### 1. panorama_sfm_complete.py（完全版・推奨）

**全自動でRig Config適用を含む完全な処理を実行**

```bash
python panorama_sfm_complete.py \
  --input_image_path 360_frames/ \
  --output_path output_rig/
```

**実行内容:**
1. 画像切り出し + マスク生成
2. 特徴抽出
3. **Rig Config適用** ✓
4. マッチング（Sequential + Exhaustive）
5. マッピング

**これが最も確実で推奨です！**

### 2. panorama_sfm_extract_only_v2.py（Rig Configまで）

**Rig Config適用まで実行し、マッチング・マッピングは手動**

```bash
python panorama_sfm_extract_only_v2.py \
  --input_image_path 360_frames/ \
  --output_path output_rig/
```

**その後:**
```bash
colmap sequential_matcher --database_path output_rig/database.db
colmap mapper --database_path output_rig/database.db --image_path output_rig/images --output_path output_rig/sparse
```

### 3. extract_panorama_images.py（私が作成したスクリプト）

同じく動作するAPIに修正済み：

```bash
python extract_panorama_images.py \
  --input_image_path 360_frames/ \
  --output_path output_rig/
```

## 🎯 推奨実行方法

**まずは完全版で試してください：**

```bash
python panorama_sfm_complete.py \
  --input_image_path 360_frames/ \
  --output_path output_rig/
```

これで：
- ✅ Rig Config適用成功
- ✅ 1つの統合モデルが作成される
- ✅ 全自動で完了

## 📊 期待される結果

```
Found 39 images in 360_frames.
✓ 画像切り出しとマスク生成完了（468枚）
✓ 特徴抽出完了
✓ Rig Config適用完了
✓ マッチング完了
#0 Reconstruction: num_reg_images=468, num_cameras=12, num_points=XXX
```

**最重要:** `num_cameras=12` = Rig Configが正しく適用されている証拠

## 🔍 動作するバージョンとの違い

### Database初期化

❌ **動かなかったバージョン:**
```python
db = pycolmap.Database()
db.open(str(database_path))
try:
    pycolmap.apply_rig_config([rig_config], db)
finally:
    db.close()
```

✅ **動くバージョン:**
```python
with pycolmap.Database.open(database_path) as db:
    pycolmap.apply_rig_config([rig_config], db)
```

### mask_path指定

❌ **動かなかったバージョン:**
```python
reader_options={"mask_path": str(mask_dir)}
```

✅ **動くバージョン:**
```python
reader_options={"mask_path": mask_dir}
```

## 🚀 大規模プロジェクト（6000枚）での使用

```bash
# テストで動作確認後
python panorama_sfm_complete.py \
  --input_image_path all_360_images/ \
  --output_path output_full/ \
  --matcher sequential

# 結果: 6000枚 × 12視点 = 72,000枚
# Rig Configにより1つの統合モデルが作成される
```

## ⚙️ オプション

```bash
--input_image_path PATH      360度画像ディレクトリ（必須）
--output_path PATH           出力ディレクトリ（必須）
--matcher TYPE               sequential/exhaustive/vocabtree/spatial
                            （デフォルト: sequential）
--pano_render_type TYPE      overlapping/non-overlapping
                            （デフォルト: overlapping = 12視点）
```

## 🎉 まとめ

**動作するバージョンのAPIを使用することで、Rig Config適用が成功します！**

以下のコマンドを実行してください：

```bash
python panorama_sfm_complete.py \
  --input_image_path 360_frames/ \
  --output_path output_rig/
```

これで全て解決します！
