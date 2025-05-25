import os
import csv

# Load nutrition table once at import
def load_nutrition(csv_path: str):
    """
    Reads a CSV of per-serving nutrition values into a dict:
    {
      food_name: {
        'calories': float,
        'protein': float,
        ...
      }, ...
    }
    Assumes first column is 'label' or 'food_name' or 'name', numeric columns follow.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Nutrition file not found: {csv_path}")
    table = {}
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # support 'label', 'food_name', or 'name' as key
            name = row.get('label') or row.get('food_name') or row.get('name')
            if not name:
                continue
            # convert numeric fields except key columns
            data = {}
            for k, v in row.items():
                if k in ('label', 'food_name', 'name'):
                    continue
                try:
                    data[k] = float(v)
                except (ValueError, TypeError):
                    data[k] = v
            table[name] = data
    return table

# Initialize nutrition table
NUTRITION_CSV = os.path.join('data', 'nutrition.csv')
_nutrition_table = load_nutrition(NUTRITION_CSV)


def estimate_calorie(food_type: str, quantity: float = 1) -> float:
    """
    Estimate total calories for a detected food based on per-serving data.

    Args:
        food_type: key matching 'label' in nutrition CSV
        quantity: number of servings
    Returns:
        total calories (float)
    """
    entry = _nutrition_table.get(food_type)
    if not entry:
        return 0.0
    calories = entry.get('calories', 0.0)
    return calories * quantity


def get_nutrition_info(food_type: str) -> dict:
    """
    Returns the full nutrition dict for a food_type per serving, or None if missing.
    """
    return _nutrition_table.get(food_type)