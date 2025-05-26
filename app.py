import pathlib
pathlib.WindowsPath = pathlib.PosixPath   
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
import hashlib
from datetime import date
from werkzeug.security import generate_password_hash, check_password_hash
from PIL import Image, ImageDraw

app = Flask(__name__)
app.secret_key = '1234567890'  # 用來加密 session，請自行修改

# 建立連線
def get_db_connection():
    return pymysql.connect(
        host = 'ai-final.c3a4yiwqg3yq.ap-southeast-2.rds.amazonaws.com',
        user = 'admin',
        password = 'admin1234',
        db = 'fitness_app',
        charset = 'utf8mb4',
        cursorclass = pymysql.cursors.DictCursor
    )


# ==== 初始化模型 ====
TRAIN_DATA_DIR_FOR_CLASSIFIER = os.path.join('data', 'images') # 根據您的實際路徑修改
classifier = FoodClassifier(
    model_path='models/efficientnet_b0_food_v2_100_classes.pt',
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
        'gender':'',
        'height_cm': '',
        'weight_kg': '',
        'age': '',
        'activity_level':''
    }

    if request.method == 'POST':
        # print("收到的表單資料:", request.form)
        form_data['username'] = request.form['username']
        form_data['gender'] = request.form['gender']
        form_data['height_cm'] = request.form['height_cm']
        form_data['weight_kg'] = request.form['weight_kg']
        form_data['age'] = request.form['age']
        form_data['activity_level'] = request.form['activity_level']

        raw_password = request.form['password']
        password_hash = hashlib.sha256(raw_password.encode('utf-8')).hexdigest()

        conn = get_db_connection()
        try:
            with conn.cursor() as cursor:
                sql = """
                    INSERT INTO users (username, password, gender, height_cm, weight_kg, age, activity_level)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """
                cursor.execute(sql, (
                    form_data['username'],
                    password_hash,
                    form_data['gender'],
                    float(form_data['height_cm']),
                    float(form_data['weight_kg']),
                    int(form_data['age']),
                    form_data['activity_level']
                ))
            conn.commit()
            return redirect(url_for('login'))
        except Exception as e:
            msg = f"帳號名稱已存在！"
            print("錯誤訊息:", e)
        finally:
            conn.close()

    return render_template('register.html', msg=msg, form_data=form_data)

@app.route('/login', methods=['GET', 'POST'])
def login():
    msg = ""

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        password_hash = hashlib.sha256(password.encode('utf-8')).hexdigest()

        conn = get_db_connection()
        with conn.cursor() as cursor:
            sql = "SELECT * FROM users WHERE username = %s"
            cursor.execute(sql, (username,))
            user = cursor.fetchone()

        conn.close()

        if user:
            if user['password'] == password_hash:
                session['user_id'] = user['id']
                session['username'] = user['username']
                return redirect(url_for('homepage'))
            else:
                session['msg'] = "密碼錯誤，請重新輸入！"
                return redirect(url_for('login'))
        else:
            session['msg'] = "使用者不存在，請確認帳號或註冊！"
            return redirect(url_for('login'))

    # 如果是 GET 請求，就從 session 取出錯誤訊息
    msg = session.pop('msg', '')  # 取出後刪除

    return render_template('login.html', msg = msg)

@app.route('/logout', methods=['POST'])
def logout():
    session.pop('username', None)
    return redirect(url_for('index'))

@app.route('/detect', methods=['POST'])
def detect():
    msg = ""
    if 'upload' in request.form: 
        # 點擊「上傳紀錄」按鈕的 POST 請求
        date_str = session.get('date')
        meal = session.get('meal')
        user_id = session.get('user_id')
        total_nutrition = session.get('total_nutrition', {})
        detections = session.get('detections', [])

        if user_id and total_nutrition:
            conn = get_db_connection()
            try:
                with conn.cursor() as cursor:
                    sql = """
                        INSERT INTO daily_nutrition 
                        (user_id, date, meal, calories, protein, fats, carbohydrates, fiber, sugars, sodium)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """
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
                    msg = "✔️ 上傳成功！紀錄已儲存到資料庫。"
            except Exception as e:
                msg = f"上傳失敗：{e}"
            finally:
                conn.close()

        return render_template('detect.html', detections=detections, date=date_str, meal=meal, msg=msg)

    else: 
        if 'image' not in request.files:
            return redirect(url_for('index'))

        file = request.files['image']
        image_path = os.path.join(UPLOAD_FOLDER, 'upload.jpg')
        file.save(image_path)

        # 取得日期、餐別
        date_str = request.form.get('date')
        meal = request.form.get('meal')

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
                if nutrition == None : 
                    nutrition =  {
                        'weight': 0.0, 
                        'calories': 0.0, 
                        'protein': 0.0, 
                        'carbohydrates': 0.0, 
                        'fats': 0.0, 
                        'fiber': 0.0, 
                        'sugars': 0.0, 
                        'sodium': 0.0}

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

        # 存到 session
        session['detections'] = results_data
        session['date'] = date_str
        session['meal'] = meal
        session['total_nutrition'] = total_nutrition

        return render_template('detect.html', detections=results_data, date=date_str, meal=meal, msg=msg)

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
    if 'user_id' not in session:
        return jsonify({'error': '未登入，請先登入帳號'}), 401

    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            # 查該日所有紀錄
            sql = """
            SELECT * FROM daily_nutrition WHERE date = %s AND user_id = %s
            """
            cursor.execute(sql, (date_str, session['user_id']))
            records = cursor.fetchall()

            # 查使用者資料
            sql2 = "SELECT * FROM users WHERE id = %s"
            cursor.execute(sql2, (session['user_id'],))
            user_record = cursor.fetchone()

            if not user_record:
                return jsonify({'error': '找不到使用者資料'}), 404

            # 修正欄位名稱
            demand = calculate_demand(
                user_record.get('height_cm', 0),
                user_record.get('weight_kg', 0),
                user_record.get('age', 0),
                user_record.get('gender', ''),
                user_record.get('activity_level', '')
            )

            if not records:
                return jsonify({'msg': '該日無攝取紀錄'})

            # 加總
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

@app.route('/profile')
def profile():
    user_id = session.get('user_id')
    if not user_id:
        return redirect(url_for('index'))

    conn = get_db_connection()
    try:
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            # 查詢基本資料
            cursor.execute("SELECT username, height_cm, weight_kg, gender, age, activity_level FROM users WHERE id = %s", (user_id,))
            user_data = cursor.fetchone()

            if not user_data:
                return "找不到使用者資料", 404

            # 計算 BMR
            if user_data['gender'] == 'male':
                bmr = 66 + (13.7 * user_data['weight_kg']) + (5 * user_data['height_cm']) - (6.8 * user_data['age'])
            else:
                bmr = 655 + (9.6 * user_data['weight_kg']) + (1.8 * user_data['height_cm']) - (4.7 * user_data['age'])

            # 計算 TDEE (假設活動係數 1.55 中等活動量)
            activity_map = {
                'less': 1.2,
                'low': 1.375,
                'medium': 1.55,
                'high': 1.725,
                'extreme_high': 1.9
            }
            activity_factor = activity_map.get(user_data['activity_level'], 1.2)

            tdee = bmr * activity_factor

            # 查詢當日總攝取營養
            import datetime
            today = datetime.date.today().strftime("%Y-%m-%d")
            cursor.execute("""
                SELECT 
                    COALESCE(SUM(calories),0) as calories,
                    COALESCE(SUM(protein),0) as protein,
                    COALESCE(SUM(fats),0) as fats,
                    COALESCE(SUM(carbohydrates),0) as carbohydrates,
                    COALESCE(SUM(fiber),0) as fiber,
                    COALESCE(SUM(sugars),0) as sugars,
                    COALESCE(SUM(sodium),0) as sodium
                FROM daily_nutrition
                WHERE user_id = %s AND date = %s
            """, (user_id, today))
            daily_nutrition = cursor.fetchone()

    finally:
        conn.close()

    return render_template('profile.html', user=user_data, bmr=bmr, tdee=tdee, daily=daily_nutrition)


if __name__ == '__main__':
    app.run(debug=True)