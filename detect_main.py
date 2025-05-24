import pathlib
pathlib.PosixPath = pathlib.WindowsPath

import os
from PIL import Image, ImageDraw

from src.classifier import FoodClassifier
from src.detector import FoodDetector
from src.calorie_estimator import estimate_calorie, get_nutrition_info
from src.recommender import recommend_by_culture
from src.ingredient_helper import ask_user_ingredients, refine_calorie
from src.utils import load_image, print_result

# ==== 初始化模型 ====
TRAIN_DATA_DIR_FOR_CLASSIFIER = os.path.join('data', 'processed', 'food101', 'train')
classifier = FoodClassifier(
    model_path='models/food_classifier.pt',
    train_dir=TRAIN_DATA_DIR_FOR_CLASSIFIER
)
detector = FoodDetector(model_path='models/best.pt')

# ==== 使用者參數（模擬）====
user_culture = 'taiwanese'
user_history = ['braised_pork_rice', 'fried_chicken']

# ==== 主流程 ====
def process_image(image_path):
    print(f"\n Processing: {image_path}")
    image = load_image(image_path)
    if image is None:
        print(f"錯誤: 無法載入圖片 {image_path}")
        return

    # === 偵測圖片中食物位置 ===
    detections = detector.detect(image_path)
    image_with_boxes = image.copy()
    draw = ImageDraw.Draw(image_with_boxes)

    if detections.empty:
        print("No food detected.")
        return

    # 顯示偵測結果
    print(f"偵測到 {len(detections)} 個物件:")
    for i, row in detections.iterrows():
        x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
        name = row['name']
        confidence = row['confidence']
        draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=3)
        label = f"{name} ({confidence:.2f})"
        draw.text((x1, y1 - 10), label, fill="red")
    image_with_boxes.show()

      # === 分類 + 營養成分輸出 ===
    print("\n開始對每個偵測到的食物進行分類和營養成分查詢：")
    for i, row in detections.iterrows():
        print(f"\n--- 物件 {i+1} ---")
        x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
        cropped = image.crop((x1, y1, x2, y2))
        food_type = classifier.predict(cropped)
        print(f" 分類為: {food_type}")

        # 取原始營養成分（每份）
        nut = get_nutrition_info(food_type)
        if nut:
            print(" 營養成分（每份）：")
            for k, v in nut.items():
                print(f"   - {k}: {v}(g)")
        else:
            print(" 無此食物的營養資料。")

        # 輸入份數並估算卡路里
        quantity = float(input(f"請輸入 '{food_type}' 的份數 (預設 1): ") or 1)
        total_cal = estimate_calorie(food_type, quantity)
        print(f" 估算卡路里: {total_cal} kcal")

       

        

# ==== 推薦系統 ====
#def recommend_food():
    #print("\nPersonalized Menu Recommendation")
    #suggestions = recommend_by_culture(user_culture, history=user_history)
    #if suggestions:
    #    print(" 尚未嘗試過：")
    #    for food in suggestions:
    #        print(f"   • {food}")
    #else:
    #    print("您已嘗試過此文化的所有菜色！")

# ==== 主程式入口 ====
if __name__ == '__main__':
    test_image_path = 'hamburger.jpg'
    process_image(test_image_path)
    #recommend_food()
