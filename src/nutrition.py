import pandas as pd

class NutritionDB:
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self._map = {r['food_name']: r.to_dict() for _, r in df.iterrows()}

    def get(self, food_name):
        return self._map.get(food_name)

# 在主程式中只做一次：
from src.nutrition import NutritionDB
nutrition_db = NutritionDB('/mnt/data/nutrition.csv')
