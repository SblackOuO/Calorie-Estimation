def ask_user_ingredients(food_type):
    print(f"What ingredients are in your {food_type}?")
    has_meat = input("Contains meat? (y/n): ")
    has_egg = input("Contains egg? (y/n): ")
    has_vegetable = input("Contains vegetables? (y/n): ")
    return {
        "meat": has_meat.lower() == "y",
        "egg": has_egg.lower() == "y",
        "vegetable": has_vegetable.lower() == "y"
    }

def refine_calorie(base_calorie, ingredients):
    delta = 0
    if "meat" in ingredients:
        delta += 100
    if "egg" in ingredients:
        delta += 70
    if "vegetable" in ingredients:
        delta += 30
    return base_calorie + delta

