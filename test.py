
"""
Predict food type and estimate calories for a single image.
Usage:
  python predict_image.py /path/to/image.jpg 
"""
import os
import argparse
from src.classifier import FoodClassifier
from src.utils import load_image, print_result
from src.calorie_estimator import estimate_calorie
from src.ingredient_helper import ask_user_ingredients, refine_calorie

# ==== Defaults (修改为自己的路径) ====
DEFAULT_MODEL_PATH = os.path.join('models', 'food_classifier.pt')
DEFAULT_TRAIN_DIR = os.path.join('data', 'processed', 'food101', 'train')


def main():
    parser = argparse.ArgumentParser(description='Predict food type and estimate calories for a single image.')
    parser.add_argument('image_path', type=str, help='Path to the food image (jpg/png).')
    parser.add_argument('--model',    type=str, default=DEFAULT_MODEL_PATH, help='Path to classifier model file.')
    parser.add_argument('--train-dir',type=str, default=DEFAULT_TRAIN_DIR,  help='Train directory for class names.')
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f'Error: Image not found: {args.image_path}')
        return
    if not os.path.isfile(args.model):
        print(f'Error: Model not found: {args.model}')
        return

    # 初始化分類器
    print(f'Loading classifier from {args.model}...')
    classifier = FoodClassifier(model_path=args.model, train_dir=args.train_dir)

    # 讀圖並預測
    img = load_image(args.image_path)
    food_type = classifier.predict(img)
    print(f'✅ Detected food: {food_type}')

    # 互動式卡路里估算
    cook_method = input(f"Enter cooking method for '{food_type}' (fried/boiled/raw): ")
    base_cal = estimate_calorie(food_type, cook_method)
    print(f'🔢 Base calorie: {base_cal} kcal')

    # 成分微調
    ingredients = ask_user_ingredients(food_type)
    final_cal = refine_calorie(base_cal, ingredients)
    print_result(food_type, final_cal)

if __name__ == '__main__':
    main()
