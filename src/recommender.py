# Example data
cultural_foods = {
    "japanese": ["sushi", "ramen", "udon"],
    "taiwanese": ["braised_pork_rice", "beef_noodle"],
    "western": ["hamburger", "pizza"]
}

def recommend_by_culture(culture, history=None):
    candidates = set(cultural_foods.get(culture, []))
    if history:
        candidates -= set(history)
    return list(candidates)
