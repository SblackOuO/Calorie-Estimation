
import os
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from src.classifier import FoodClassifier

def main():
    # ---- 參數設定 ----
    TEST_DIR   = os.path.join('data', 'processed', 'food101', 'test')
    TRAIN_DIR  = os.path.join('data', 'processed', 'food101', 'train')
    MODEL_PATH = os.path.join('models', 'food_classifier.pt')
    BATCH_SIZE = 64

    print('=== Starting evaluation on test set ===')
    # 1. 載入已訓練好的分類器
    classifier = FoodClassifier(model_path=MODEL_PATH, train_dir=TRAIN_DIR)
    model      = classifier.model
    device     = next(model.parameters()).device
    model.eval()  # 切到 eval 模式

    # 2. 前處理（若您的 FoodClassifier 內已有 val_transform，就用它；否則自行定義）
    if hasattr(classifier, 'val_transform'):
        transform = classifier.val_transform
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    # 3. 準備 Test Dataset & DataLoader
    test_dataset = datasets.ImageFolder(TEST_DIR, transform=transform)
    test_loader  = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )

    # 4. 推論並收集結果
    y_true, y_pred = [], []
    misclassified  = []

    for imgs, labels in tqdm(test_loader, desc='Evaluating batches', unit='batch'):
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.no_grad():
            outputs = model(imgs)
        preds = outputs.argmax(dim=1)

        for i, (t, p) in enumerate(zip(labels, preds)):
            true_label = test_dataset.classes[t]
            pred_label = test_dataset.classes[p]
            y_true.append(true_label)
            y_pred.append(pred_label)
            if pred_label != true_label:
                # 對應到原始檔案路徑
                path, _ = test_dataset.samples[len(y_true)-1]
                misclassified.append((true_label, pred_label, path))

    # 5. 列印 Classification Report
    print("\n=== Classification Report ===")
    print(classification_report(
        y_true,
        y_pred,
        labels=test_dataset.classes,
        zero_division=0
    ))

    # 6. 列印 Confusion Matrix
    print("=== Confusion Matrix ===")
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=test_dataset.classes
    )
    print(cm)

    # 7. 最易混淆的前 10 對
    print("\n=== Top 10 Confusion Pairs ===")
    for (t, p), cnt in Counter((t, p) for t, p, _ in misclassified).most_common(10):
        print(f"  {t} -> {p}: {cnt} cases")

    # 8. 範例錯誤案例
    print("\n=== Sample Misclassified Images ===")
    for true, pred, path in misclassified[:5]:
        print(f"  True: {true}, Pred: {pred}  →  {path}")

    print('\n=== Evaluation complete ===')

if __name__ == "__main__":
    main()
