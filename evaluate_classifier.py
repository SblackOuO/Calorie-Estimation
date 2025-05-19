import os
from collections import Counter
from src.classifier import FoodClassifier
from src.utils import load_image
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm


def evaluate(test_dir, model_path, train_dir):
    # 1. 載入模型
    print(f">>> Loading classifier from {model_path}")
    classifier = FoodClassifier(model_path=model_path, train_dir=train_dir)

    # 2. 準備測試集路徑與類別
    classes = sorted([d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))])
    print(f">>> Found {len(classes)} classes in test set")

    y_true, y_pred = [], []
    misclassified = []

    # 3. 逐類別評估
    for cls in tqdm(classes, desc='Classes', unit='class'):
        cls_dir = os.path.join(test_dir, cls)
        images = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.png'))]
        print(f"\n>>> Evaluating class '{cls}' ({len(images)} images)")
        for img in tqdm(images, desc=f'  Processing {cls}', leave=False, unit='img'):
            path = os.path.join(cls_dir, img)
            pred = classifier.predict(load_image(path))
            y_true.append(cls)
            y_pred.append(pred)
            if pred != cls:
                misclassified.append((cls, pred, path))

    # 4. 輸出報告
    labels = classifier.class_names
    print("\n=== Classification Report ===")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))

    print("=== Confusion Matrix ===")
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print(cm)

    # 5. 顯示最易混淆的前 10 對
    pair_counts = Counter((t, p) for t,p,_ in misclassified)
    print("\n=== Top 10 Confusion Pairs ===")
    for (t,p), cnt in pair_counts.most_common(10):
        print(f"  {t} -> {p}: {cnt} instances")

    # 6. 顯示部分錯誤案例路徑
    print("\n=== Sample Misclassified Images ===")
    for true, pred, path in misclassified[:5]:
        print(f"  True: {true}, Pred: {pred} -> {path}")


if __name__ == "__main__":
    TEST_DIR   = os.path.join('data','processed','food101','test')
    MODEL_PATH = os.path.join('models','food_classifier.pt')
    TRAIN_DIR  = os.path.join('data','processed','food101','train')
    print('=== Starting evaluation on test set ===')
    evaluate(TEST_DIR, MODEL_PATH, TRAIN_DIR)
    print('=== Evaluation complete ===')
