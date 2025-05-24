import os
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from tqdm import tqdm

from src.calorie_estimator import estimate_calorie
from src.recommender import recommend_by_culture
from src.ingredient_helper import ask_user_ingredients, refine_calorie
from src.utils import load_image, print_result
from src.classifier import FoodClassifier

# ==== 0. 資料切分（若尚未切分則自動執行） ====
RAW_DIR      = os.path.join('data', 'food101', 'food-101', 'images')
PROCESSED_DIR = os.path.join('data', 'processed', 'food101')
TRAIN_DIR     = os.path.join(PROCESSED_DIR, 'train')
VAL_DIR       = os.path.join(PROCESSED_DIR, 'val')
TEST_DIR      = os.path.join(PROCESSED_DIR, 'test')
TRAIN_RATIO   = 0.8
VAL_RATIO     = 0.1
SEED          = 42

if not os.path.isdir(TRAIN_DIR) or not os.listdir(TRAIN_DIR):
    print('>>> Splitting Food-101 dataset into train/val/test...')
    try:
        from split_food101 import split_dataset
        split_dataset(RAW_DIR, PROCESSED_DIR, TRAIN_RATIO, VAL_RATIO, SEED)
    except ImportError:
        raise ImportError('找不到 split_food101.py，請確認該檔案存在於專案根目錄')

# ==== 1. 裝置與模型路徑 ====
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = os.path.join('models', 'food_classifier.pt')

# ==== 2. 使用者設定 ====
user_culture = 'taiwanese'
user_history = []  # 可填入已嘗試的菜名

# ==== 3. 訓練分類器（含增強、正則、調度、早停、AMP） ====
def train_classifier(epochs=10, batch_size=128, patience=5, lr=5e-5):
    print('>>> Start training classifier on', DEVICE)
    # 資料增強：微調強度
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),                    # 降低旋轉角度至 ±10°
        transforms.ColorJitter(0.15, 0.15, 0.15, 0.05),    # 收斂亮度/對比度
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
        transforms.RandomErasing(p=0.05, scale=(0.02,0.33), ratio=(0.3,3.3))  # 降低遮蔽機率
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    train_set = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
    val_set   = datasets.ImageFolder(VAL_DIR,   transform=val_transform)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader   = torch.utils.data.DataLoader(
        val_set,   batch_size=batch_size, shuffle=False, num_workers=4)

    num_classes = len(train_set.classes)
    model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.last_channel, num_classes)
    )
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    scaler = GradScaler() if DEVICE=='cuda' else None
    best_acc = 0.0
    no_improve = 0

    for epoch in range(epochs):
        print(f'>>> Epoch {epoch+1}/{epochs}')
        # 訓練
        model.train()
        train_loss, train_total, train_correct = 0.0, 0, 0
        for imgs, labels in tqdm(train_loader, desc=' Training', unit='batch'):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            train_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            train_total += labels.size(0)
            train_correct += (preds==labels).sum().item()
        train_loss /= train_total
        train_acc  = train_correct / train_total
        print(f'    Train loss: {train_loss:.4f}, acc: {train_acc:.4f}')

        # 驗證
        model.eval()
        val_loss, val_total, val_correct = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=' Validation', unit='batch'):
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                if scaler:
                    with torch.cuda.amp.autocast():
                        outputs = model(imgs)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)

                val_loss += loss.item() * labels.size(0)
                preds = outputs.argmax(dim=1)
                val_total += labels.size(0)
                val_correct += (preds==labels).sum().item()
        val_loss /= val_total
        val_acc  = val_correct / val_total
        print(f'    Val   loss: {val_loss:.4f}, acc: {val_acc:.4f}')

        scheduler.step()
        if val_acc > best_acc:
            best_acc, no_improve = val_acc, 0
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            torch.save(model.state_dict(), MODEL_PATH)
            print(f'    Saved best model (val_acc={best_acc:.4f})')
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f'No improvement for {patience} epochs. Early stopping.')
                break

    print('>>> Training complete. Best val accuracy:', best_acc)
    return model

# ==== 4. 確保模型存在 ====
def ensure_model():
    print('>>> Checking classifier model...')
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        train_classifier()
    else:
        print('    Model file found:', MODEL_PATH)

# ==== 5. 圖片處理流程 ====
def process_image(image_path, classifier):
    print('>>> Processing image:', image_path)
    img = load_image(image_path)
    food_type = classifier.predict(img)
    print('    Detected:', food_type)
    cook_method = input(f"    Cooking method for '{food_type}'? ")
    base_cal = estimate_calorie(food_type, cook_method)
    print(f'    Base calorie: {base_cal} kcal')
    ing = ask_user_ingredients(food_type)
    final_cal = refine_calorie(base_cal, ing)
    print_result(food_type, final_cal)

# ==== 6. 推薦流程 ====
def recommend_food():
    print('>>> Generating recommendations')
    suggestions = recommend_by_culture(user_culture, history=user_history)
    if suggestions:
        print('    You haven\'t tried:')
        for f in suggestions:
            print('     -', f)
    else:
        print('    All dishes tried')

# ==== 7. 主程式入口 ====
if __name__ == '__main__':
    print('=== Main Pipeline Start ===')
    print('Device in use:', DEVICE)
    print('Train:', TRAIN_DIR)
    print('Val:  ', VAL_DIR)
    print('Test:', TEST_DIR)

    ensure_model()
    classifier = FoodClassifier(model_path=MODEL_PATH, train_dir=TRAIN_DIR)

    print('\n>>> Starting inference on test set')
    if os.path.isdir(TEST_DIR):
        files = [f for f in os.listdir(TEST_DIR) if f.lower().endswith(('.jpg','.png'))]
        for i, fn in enumerate(files, 1):
            print(f'Inference {i}/{len(files)}')
            process_image(os.path.join(TEST_DIR, fn), classifier)
    else:
        print('    No test data found at', TEST_DIR)

    print('\n=== Inference Complete ===')
    recommend_food()
    print('=== Pipeline Finished ===')
