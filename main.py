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
from src.detector import FoodDetector

# ==== 1. 資料夾設定（已切分） ====
PROCESSED_DIR = os.path.join('data', 'processed', 'food101')
TRAIN_DIR     = os.path.join(PROCESSED_DIR, 'train')
VAL_DIR       = os.path.join(PROCESSED_DIR, 'val')
TEST_DIR      = os.path.join(PROCESSED_DIR, 'test')

# ==== 2. 裝置與模型路徑 ====
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = os.path.join('models', 'food_classifier.pt')

# ==== 3. 使用者設定 ====
user_culture = 'taiwanese'
user_history = []  # 可填入已嘗試的菜名

# ==== 4. 訓練分類器（不含資料預處理） ====
def train_classifier(epochs=20, batch_size=64):
    print('>>> Start training classifier on', DEVICE)
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    train_set = datasets.ImageFolder(TRAIN_DIR, transform=transform)
    val_set   = datasets.ImageFolder(VAL_DIR,   transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader   = torch.utils.data.DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=4)

    num_classes = len(train_set.classes)
    model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    best_acc = 0.0

    for epoch in range(epochs):
        print(f'>>> Epoch {epoch+1}/{epochs}')
        # Training phase with progress bar
        model.train()
        train_bar = tqdm(train_loader, desc=' Training', unit='batch')
        total, correct = 0, 0
        for imgs, labels in train_bar:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            preds = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            train_acc = correct / total
            train_bar.set_postfix(loss=f'{loss.item():.4f}', acc=f'{train_acc:.4f}')

                # Validation phase with progress bar
        model.eval()
        val_bar = tqdm(val_loader, desc=' Validation', unit='batch')
        val_total, val_correct = 0, 0
        with torch.no_grad():
            for imgs, labels in val_bar:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                preds = outputs.argmax(dim=1)
                val_total += labels.size(0)
                val_correct += (preds == labels).sum().item()
                val_acc = val_correct / val_total
                val_bar.set_postfix(acc=f'{val_acc:.4f}')

        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            torch.save(model.state_dict(), MODEL_PATH)
            best_acc = val_acc
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            torch.save(model.state_dict(), MODEL_PATH)

    print('>>> Training complete. Best val accuracy:', best_acc)
    return model

# ==== 5. 確保模型存在 ====
def ensure_model():
    print('>>> Checking classifier model...')
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        train_classifier()
    else:
        print('    Model file found:', MODEL_PATH)

# ==== 6. 圖片處理流程 ====
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

# ==== 7. 推薦流程 ====
def recommend_food():
    print('>>> Generating recommendations')
    suggestions = recommend_by_culture(user_culture, history=user_history)
    if suggestions:
        print('    You haven\'t tried:')
        for f in suggestions:
            print('     -', f)
    else:
        print('    All dishes tried')

# ==== 8. 主程式入口 ====
if __name__ == '__main__':
    print('=== Main Pipeline Start ===')
    print('Device in use:', DEVICE)
    print('Train:', TRAIN_DIR)
    print('Val:  ', VAL_DIR)
    print('Test: ', TEST_DIR)

    ensure_model()
    classifier = FoodClassifier(model_path=MODEL_PATH, train_dir=TRAIN_DIR)
    detector = FoodDetector(model_path = 'model/yolov5_food.pt')

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
