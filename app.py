from flask import Flask, jsonify, render_template, request, redirect, url_for, session
from src.classifier import FoodClassifier
from src.detector import FoodDetector
from src.calorie_estimator import estimate_calorie, get_nutrition_info
from src.recommender import recommend_by_culture
from src.ingredient_helper import ask_user_ingredients, refine_calorie
from src.utils import load_image, print_result, calculate_demand
import shutil
import os
import pymysql
from datetime import date
from werkzeug.security import generate_password_hash, check_password_hash
from PIL import Image, ImageDraw

app = Flask(__name__)
app.secret_key = '1234567890'  # 用來加密 session，請自行修改

db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '0425',        # 依你的密碼而定
    'database': 'calorie_ai',
    'charset': 'utf8mb4',
    'cursorclass': pymysql.cursors.DictCursor
}

# 建立連線
def get_db_connection():
    return pymysql.connect(**db_config)

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

@app.route('/homepage')
def homepage():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('homepage.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    msg = ""
    form_data = {
        'username': '',
        'sex':'',
        'height': '',
        'weight': '',
        'age': '',
        'activity_level':''
    }

    if request.method == 'POST':
        form_data['username'] = request.form['username']
        form_data['sex'] = request.form['sex']
        form_data['height'] = request.form['height']
        form_data['weight'] = request.form['weight']
        form_data['age'] = request.form['age']
        form_data['activity_level'] = request.form['activity_level']

        password = generate_password_hash(request.form['password'])

        conn = get_db_connection()
        try:
            with conn.cursor() as cursor:
                sql = "INSERT INTO users (username, password, sex, height, weight, age, activity_level) VALUES (%s, %s, %s, %s, %s, %s, %s)"
                cursor.execute(sql, (
                    form_data['username'],
                    password,
                    form_data['sex'],
                    float(form_data['height']),
                    float(form_data['weight']),
                    int(form_data['age']),
                    form_data['activity_level']
                ))
            conn.commit()
            return redirect(url_for('login'))
        except Exception as e:
            msg = f"註冊失敗：帳號名稱已存在"
        finally:
            conn.close()

    return render_template('register.html', msg=msg, form_data=form_data)




@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = get_db_connection()
        with conn.cursor() as cursor:
            sql = "SELECT * FROM users WHERE username = %s"
            cursor.execute(sql, (username,))
            user = cursor.fetchone()

        conn.close()

        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            return redirect(url_for('homepage'))
        else:
            return "登入失敗，帳號或密碼錯誤"

    return render_template('login.html')


@app.route('/logout', methods=['POST'])
def logout():
    session.pop('username', None)
    return redirect(url_for('index'))


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

     # 初始總攝取量
    total_nutrition = {
        'calories': 0,
        'protein': 0,
        'fats': 0,
        'carbohydrates': 0,
        'fiber': 0,
        'sugars': 0,
        'sodium': 0
    }

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

            for key in total_nutrition:
                total_nutrition[key] += nutrition.get(key, 0)

            results_data.append({
                'index': i+1,
                'name': food_type,
                'base_cal': base_cal,
                'nutrition': nutrition,
                'crop_image': crop_filename
            })

        image.save(os.path.join(UPLOAD_FOLDER, 'result.jpg'))

        date_str = request.form.get('date')  # 格式應為 'YYYY-MM-DD'
        meal = request.form.get('meal')      # 早餐、午餐、晚餐、其他

        # 儲存每日攝取資料到資料庫
        user_id = session.get('user_id')
        if user_id:
            conn = get_db_connection()
            try:
                with conn.cursor() as cursor:
                    sql = """
                        INSERT INTO daily_nutrition 
                        (user_id, date, meal, calories, protein, fats, carbohydrates, fiber, sugars, sodium)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """
                    #today = date.today()
                    cursor.execute(sql, (
                        user_id, date_str, meal,
                        total_nutrition['calories'],
                        total_nutrition['protein'],
                        total_nutrition['fats'],
                        total_nutrition['carbohydrates'],
                        total_nutrition['fiber'],
                        total_nutrition['sugars'],
                        total_nutrition['sodium']
                    ))
                    conn.commit()
            except Exception as e:
                print("紀錄攝取量失敗：", e)
            finally:
                conn.close()
    return render_template('detect.html', detections=results_data)

@app.route('/calendar')
def calendar():
    return render_template('calendar.html')  # 你先前的日曆前端頁面

@app.route('/calendar-data')
def calendar_data():
    # 取得 FullCalendar 要求的時間範圍參數
    start = request.args.get('start')  # yyyy-mm-dd
    end = request.args.get('end')      # yyyy-mm-dd

    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            sql = """
            SELECT date, SUM(calories) AS total_calories
            FROM daily_nutrition
            WHERE (date BETWEEN %s AND %s) AND user_id = %s
            GROUP BY date
            """
            cursor.execute(sql, (start, end, session['user_id']))
            rows = cursor.fetchall()
            
            events = []
            for row in rows:
                events.append({
                    'title': f"熱量 {row['total_calories']} kcal",
                    'start': row['date'].strftime('%Y-%m-%d'),
                    'allDay': True,
                })
            return jsonify(events)
    finally:
        conn.close()

@app.route('/calendar-detail')
def calendar_detail():
    date_str = request.args.get('date')  # yyyy-mm-dd
    if not date_str:
        return jsonify({'error': '缺少日期參數'}), 400

    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            sql = """
            SELECT * FROM daily_nutrition as d WHERE d.date = %s and d.user_id = %s
            """
            cursor.execute(sql, (date_str, session['user_id']))
            records = cursor.fetchall()

            sql2 = """
            SELECT * FROM users WHERE id = %s
            """
            cursor.execute(sql2, (session['user_id'],))
            user_record = cursor.fetchone()
            demand = calculate_demand(user_record.get('height',0), user_record.get('weight',0), user_record.get('age',0), user_record.get('sex',0), user_record.get('activity_level',0))

            if not records:
                return jsonify({'msg': '該日無攝取紀錄'})
            
            

            # 你可在這邊計算營養建議判斷
            # 這裡簡單示範：若蛋白質 < 50g 就建議補充蛋白質
            calories_sum = sum(r.get('calories', 0) for r in records)
            protein_sum = sum(r.get('protein', 0) for r in records)
            fats_sum = sum(r.get('fats', 0) for r in records)
            carbohydrates_sum = sum(r.get('carbohydrates', 0) for r in records)
            fiber_sum = sum(r.get('fiber', 0) for r in records)
            sugars_sum = sum(r.get('sugars', 0) for r in records)
            sodium_sum = sum(r.get('sodium', 0) for r in records)

            suggestions = []
            if calories_sum < demand['calories_demand']:
                suggestions.append('熱量攝取不足，建議增加整體飲食攝取量，例如全穀、堅果等高熱量健康食物。')
            if protein_sum < demand['protein_demand']:
                suggestions.append('蛋白質攝取不足，建議增加豆魚蛋肉類、乳製品等。')
            if fats_sum < demand['fats_demand']:
                suggestions.append('脂肪攝取不足，建議可補充堅果、酪梨、橄欖油等健康脂肪。')
            if carbohydrates_sum < demand['carbohydrates_demand']:
                suggestions.append('碳水化合物攝取不足，建議補充糙米、燕麥、地瓜等全穀類食物。')
            if fiber_sum < demand['fiber_demand']:
                suggestions.append('膳食纖維攝取不足，建議增加蔬菜、水果與全穀攝取。')
            if sugars_sum < demand['sugars_demand']:
                suggestions.append('糖攝取不足，可適量食用水果或乳製品。')
            if sodium_sum < demand['sodium_demand']:
                suggestions.append('鈉攝取不足，可適量攝取鹽分或含鈉調味料，但注意不要過量。')

            return jsonify({
                'date': date_str,
                'records': records,
                'suggestions': suggestions
            })
    finally:
        conn.close()


if __name__ == '__main__':
    app.run(debug=True)