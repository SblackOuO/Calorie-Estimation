
"""
Split Food-101 dataset into train/val/test directories with hardcoded parameters.
"""
import os
import shutil
import random

# ==== 參數設定（直接在此修改） ====
# 請設定為解壓後的實際路徑
RAW_DIR     = os.path.join('data', 'food101', 'food-101', 'images')
OUT_DIR     = os.path.join('data', 'processed', 'food101')
TRAIN_RATIO = 0.8
VAL_RATIO   = 0.1
SEED        = 42

# ==== 切分函式 ====
def split_dataset(raw_dir, out_dir, train_ratio, val_ratio, seed=42):
    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(f"Raw images folder not found: {raw_dir}")
    test_ratio = 1.0 - train_ratio - val_ratio
    assert 0 < train_ratio < 1 and 0 <= val_ratio < 1 and test_ratio > 0, \
        "Ratios must be between 0 and 1 and sum to 1"
    # 建立 split 資料夾
    for split in ('train', 'val', 'test'):
        d = os.path.join(out_dir, split)
        if os.path.isdir(d):
            shutil.rmtree(d)
        os.makedirs(d)
    random.seed(seed)
    # 讀取所有 class
    classes = sorted([d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))])
    for cls in classes:
        cls_src = os.path.join(raw_dir, cls)
        images = [f for f in os.listdir(cls_src)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(images)
        n = len(images)
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)
        splits = {
            'train': images[:n_train],
            'val':   images[n_train:n_train + n_val],
            'test':  images[n_train + n_val:]
        }
        for split, imgs in splits.items():
            dst_dir = os.path.join(out_dir, split, cls)
            os.makedirs(dst_dir, exist_ok=True)
            for img in imgs:
                shutil.copy2(
                    os.path.join(cls_src, img),
                    os.path.join(dst_dir, img)
                )
    print(f"Split complete → train {train_ratio}, val {val_ratio}, test {test_ratio}")

# ==== 主程式執行 ====
if __name__ == '__main__':
    split_dataset(RAW_DIR, OUT_DIR, TRAIN_RATIO, VAL_RATIO, SEED)
