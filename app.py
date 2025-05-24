from flask import Flask, render_template, request, redirect, url_for
from src.classifier import FoodClassifier
from src.detector import FoodDetector
from src.calorie_estimator import estimate_calorie, get_nutrition_info
from src.recommender import recommend_by_culture
from src.ingredient_helper import ask_user_ingredients, refine_calorie
from src.utils import load_image, print_result
import shutil
import os
from PIL import Image, ImageDraw

app = Flask(__name__)
# ==== 初始化模型 ====
TRAIN_DATA_DIR_FOR_CLASSIFIER = os.path.join('data', 'images') # 根據您的實際路徑修改
classifier = FoodClassifier(
    model_path='models/food_classifier.pt',
    train_dir=TRAIN_DATA_DIR_FOR_CLASSIFIER # 提供這個參數
)
detector = FoodDetector(model_path='models/best.pt')

UPLOAD_FOLDER = 'static'
CROP_FOLDER = os.path.join(UPLOAD_FOLDER, 'crops')
os.makedirs(CROP_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return redirect(url_for('index'))

    file = request.files['image']
    image_path = os.path.join(UPLOAD_FOLDER, 'upload.jpg')
    file.save(image_path)

    # 用 YOLO 模型做偵測
    detections = detector.detect(image_path)
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    results_data = []

    # 清空舊的裁切圖
    shutil.rmtree(CROP_FOLDER)
    os.makedirs(CROP_FOLDER)

    if not detections.empty:
        for i, row in detections.iterrows():
            x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
            name = row['name']
            conf = float(row['confidence'])

            draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=3)
            draw.text((x1, max(y1-10, 0)), f"{name} ({conf:.2f})", fill="red")

            crop = image.crop((x1, y1, x2, y2))
            crop_filename = f"crop_{i}.jpg"
            crop_path = os.path.join(CROP_FOLDER, crop_filename)
            crop.save(crop_path)
            food_type = classifier.predict(crop)
            
            base_cal = estimate_calorie(food_type)
            nutrition = get_nutrition_info(food_type)

            results_data.append({
                'index': i+1,
                'name': food_type,
                'base_cal': base_cal,
                'nutrition': nutrition,
                'crop_image': crop_filename
            })

        image.save(os.path.join(UPLOAD_FOLDER, 'result.jpg'))
    return render_template('detect.html', detections=results_data)


if __name__ == '__main__':
    app.run(debug=True)