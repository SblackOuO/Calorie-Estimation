import pathlib
pathlib.WindowsPath = pathlib.PosixPath   
import os
from PIL import Image, ImageDraw

from src.classifier import FoodClassifier
from src.detector import FoodDetector
from src.calorie_estimator import estimate_calorie
from src.recommender import recommend_by_culture
from src.ingredient_helper import ask_user_ingredients, refine_calorie
from src.utils import load_image, print_result
#opencv-python seaborn
# ==== 初始化模型 ====
TRAIN_DATA_DIR_FOR_CLASSIFIER = os.path.join('data', 'images') # 根據您的實際路徑修改
classifier = FoodClassifier(
    model_path='models/food_classifier.pt',
    train_dir=TRAIN_DATA_DIR_FOR_CLASSIFIER # 提供這個參數
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
    
    if not detections.empty:
        print(f"偵測到 {len(detections)} 個物件:")
        for i, row in detections.iterrows():
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            confidence = row['confidence']
            name = row['name']
            
            draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=3)
            label = f"{name} ({confidence:.2f})"
            text_position = (x1, y1 - 10 if y1 - 10 > 0 else y1 + 2)
            try:
                draw.text(text_position, label, fill="red")
            except Exception as e:
                print(f"繪製文字時發生錯誤: {e} (可能是缺少字體)")
                draw.text(text_position, name, fill="red")
            
            print(f"  - 物件 {i}: {name} (信心度: {confidence:.2f}) at [{x1},{y1},{x2},{y2}]")
        
        image_with_boxes.show()
    if detections.empty:
        print("No food detected.")
        return

    print("\n開始對每個偵測到的食物進行分類和熱量估算：")
    for i, row in detections.iterrows():
        print(f"\n--- 正在處理偵測到的物件 {i+1} --- ")
        x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
        cropped = image.crop((x1, y1, x2, y2))
        food_type = classifier.predict(cropped)
        print(f" 由 FoodClassifier 分類為: {food_type}")

        cook_method = input(f" What is the cooking method for '{food_type}' (YOLO name: {row['name']})? (fried/boiled/raw): ")

        base_cal = estimate_calorie(food_type, cook_method)
        print(f" Base calorie: {base_cal} kcal")

        ing = ask_user_ingredients(food_type)
        final_cal = refine_calorie(base_cal, ing)

        print_result(food_type, final_cal)

# ==== 餐點推薦 ====
def recommend_food():
    print("\n Personalized Menu Recommendation")
    suggestions = recommend_by_culture(user_culture, history=user_history)
    if suggestions:
        print(" You haven't tried:")
        for food in suggestions:
            print(f"   {food}")
    else:
        print("You've tried everything in this cuisine!")

# ==== 主程式入口 ====
if __name__ == "__main__":
    test_image_path = 'hamburger_donut.jpeg'
    process_image(test_image_path)

    recommend_food()