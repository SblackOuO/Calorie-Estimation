import os
from PIL import Image

def load_image(img_path):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"{img_path} not found.")
    return Image.open(img_path)

def print_result(food, cal):
    print(f"Food: {food}, Estimated Calories: {cal} kcal")

def calculate_demand(heights, weights, ages, sex, activity_level):
    if sex == 'male':
        s = 5
    elif sex == 'female':
        s = -161
        
    BMR= 10 * weights + 6.25 * heights - 5 * ages + s
    if activity_level == 'less':
        act = 1.2
    elif activity_level == 'low':
        act = 1.375
    elif activity_level == 'medium':
        act = 1.55
    elif activity_level == 'high':
        act = 1.725
    elif activity_level == 'extreme':
        act = 1.9
        
    TDEE = BMR * act
    protein_demand = TDEE * 0.15 / 4
    fats_demand = TDEE * 0.25 / 9
    carbohydrates_demand = TDEE * 0.55 / 4
    sugars_demand = TDEE * 0.1 / 4
    sodium_demand = 0
    fiber_demand = 0
    if(1 <= ages < 3):
        sodium_demand = 1000
    elif(3 <= ages <= 5):
        sodium_demand = 1100
    elif(6 <= ages <= 9):
        sodium_demand = 1500
    elif(10 <= ages <= 12):
        sodium_demand = 1900
    else:
        sodium_demand = 2200

    if sex == 'male':
        if(1 <= ages <= 3):
            fiber_demand = 14
        elif(4 <= ages <= 8):
            fiber_demand = 19
        elif(9 <= ages <= 13):
            fiber_demand = 31
        elif(14 <= ages <= 18):
            fiber_demand = 38
        elif(19 <= ages <= 50):
            fiber_demand = 38
        elif(51 <= ages):
            fiber_demand = 30
    elif sex == 'female':
        if(1 <= ages <= 3):
            fiber_demand = 14
        elif(4 <= ages <= 8):
            fiber_demand = 19
        elif(9 <= ages <= 13):
            fiber_demand = 26
        elif(14 <= ages <= 18):
            fiber_demand = 26
        elif(19 <= ages <= 50):
            fiber_demand = 25
        elif(51 <= ages):
            fiber_demand = 21
    

    result = {'calories_demand' : TDEE,
              'protein_demand' : protein_demand,
              'fats_demand' : fats_demand,
              'carbohydrates_demand' : carbohydrates_demand,
              'sugars_demand' : sugars_demand,
              'sodium_demand' : sodium_demand,
              'fiber_demand' : fiber_demand }
    
    return result
    


