"""Harness for AI-generated 'Sloptails' (multi-step, rail-guided).

The generation is split into small, constrained steps so a small local model
stays on the rails:
  1. pick a cocktail *class* (template) from cocktail_classes.json
  2. for every role in that class (spirit / sweet / sour / bitter / filler)
     pick ONE ingredient, shown only the available ingredients of that category
  3. amounts come from the class ratios (not the model)
  4. a final step invents a name + description

Everything model-agnostic and testable lives here; the async HTTP orchestration
lives in main.py.
"""
import os

# Configurable so nothing is hard-coded when the model changes.
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")

# Categories an ingredient can serve, in display / build order.
CATEGORIES = ["spirit", "sweet", "sour", "bitter", "filler"]

# 1 ratio unit = this many ml; ice added per glass to serve over.
BASE_UNIT_ML = 25
ICE_GRAMS = {"Longdrink": 100, "Wine": 40, "Shot": 0}

# Selectable tasting notes (English keys drive the prompt; small models cope
# better with English). German labels are shown in the UI.
TASTE_NOTES = [
    "citrus", "berry", "pomegranate", "tropical", "apple", "stone fruit",
    "juniper", "herbal", "floral", "mint", "grassy",
    "sweet", "vanilla", "caramel", "chocolate", "honey",
    "sour", "fresh", "bright", "dry",
    "bitter", "warm spice", "peppery", "smoky", "earthy", "agave",
    "molasses", "nutty", "creamy",
]
TASTE_NOTES_DE = {
    "citrus": "Zitrus", "berry": "Beere", "pomegranate": "Granatapfel",
    "tropical": "Tropisch", "apple": "Apfel", "stone fruit": "Steinobst",
    "juniper": "Wacholder", "herbal": "Kräuter", "floral": "Blumig",
    "mint": "Minze", "grassy": "Grasig",
    "sweet": "Süß", "vanilla": "Vanille", "caramel": "Karamell",
    "chocolate": "Schokolade", "honey": "Honig",
    "sour": "Sauer", "fresh": "Frisch", "bright": "Spritzig", "dry": "Trocken",
    "bitter": "Bitter", "warm spice": "Warme Gewürze", "peppery": "Pfeffrig",
    "smoky": "Rauchig", "earthy": "Erdig", "agave": "Agave",
    "molasses": "Melasse", "nutty": "Nussig", "creamy": "Cremig",
}
TASTE_EMOJI = {
    "citrus": "🍋", "berry": "🫐", "pomegranate": "🍎", "tropical": "🍍",
    "apple": "🍏", "stone fruit": "🍑", "juniper": "🌲", "herbal": "🌿",
    "floral": "🌸", "mint": "🍃", "grassy": "🌱", "sweet": "🍬",
    "vanilla": "🍦", "caramel": "🍮", "chocolate": "🍫", "honey": "🍯",
    "sour": "😝", "fresh": "💧", "bright": "✨", "dry": "🏜️",
    "bitter": "🍵", "warm spice": "🔥", "peppery": "🌶️", "smoky": "💨",
    "earthy": "🍄", "agave": "🌵", "molasses": "🫙", "nutty": "🥜", "creamy": "🥛",
}

def taste_note_options():
    """(key, german_label, emoji) triples for the UI."""
    return [(k, TASTE_NOTES_DE.get(k, k), TASTE_EMOJI.get(k, "🍸")) for k in TASTE_NOTES]


# --- Step 1: the cocktail class is picked at random (see main.py) ------------


# --- Step 2: pick one ingredient for a role ----------------------------------
def build_ingredient_messages(taste_notes, category, candidates):
    """candidates: list of ingredient dicts. Model picks one by number."""
    lines = [f'{pos + 1}. {i["name"]} — {", ".join(i.get("taste", [])) or "neutral"}'
             for pos, i in enumerate(candidates)]

    system = ("You are a cocktail expert. Choose the single ingredient that best "
              "matches the desired taste for the given role. Respond only with valid JSON.")
    user = (f"Desired taste notes: {', '.join(taste_notes) or 'surprise me'}\n"
            f"Role in the drink: {category}\n\n"
            f"Available {category} ingredients:\n" + "\n".join(lines) +
            '\n\nRespond exactly: {"ingredient": <number>}')
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# --- Step 4: invent a name + description -------------------------------------
def build_name_messages(class_name, chosen_names, taste_notes):
    system = ("You are a creative bartender. Invent a fun, original ENGLISH cocktail name "
              "and a one-sentence ENGLISH description. Respond only with valid JSON.")
    user = (f"Template: {class_name}\n"
            f"Ingredients: {', '.join(chosen_names)}\n"
            f"Taste: {', '.join(taste_notes) or 'surprise'}\n\n"
            'Respond exactly: {"name": "<creative english name>", "description": "<one english sentence>"}')
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_choice(raw, key, n):
    """Read a 1-based number from the model's JSON. Returns index in [1, n] or None."""
    if not isinstance(raw, dict):
        return None
    val = raw.get(key)
    if isinstance(val, bool):
        return None
    try:
        idx = int(val)
    except (TypeError, ValueError):
        return None
    return idx if 1 <= idx <= n else None


def candidates_for(category, ingredients, alcohol_free=False):
    """Available ingredients whose primary category matches.

    For the spirit role the alcohol_free flag decides which base is offered:
    only 0% spirits when set, only alcoholic spirits otherwise. Other roles
    (sweet/sour/filler) are non-alcoholic anyway and stay unfiltered.
    """
    cands = [i for i in ingredients
             if i.get("cat") == category and i.get("available", True)]
    if category == "spirit":
        if alcohol_free:
            cands = [i for i in cands if i.get("alc", 0) == 0]
        else:
            cands = [i for i in cands if i.get("alc", 0) > 0]
    return cands


def compute_amounts(class_def, chosen):
    """chosen: {category: ingredient dict}. Amounts come from the class ratios.

    Returns a recipe dict {glass, ingredients:[{id, amount}]} (ice prepended).
    """
    ratios = class_def.get("ratios", {})
    glass = class_def.get("glass", "Longdrink")

    ingredients = []
    ice_g = ICE_GRAMS.get(glass, 0)
    if ice_g > 0:
        ingredients.append({"id": "ice", "amount": ice_g})

    for category in CATEGORIES:  # stable, sensible order
        ing = chosen.get(category)
        if not ing:
            continue
        amount = int(round(ratios.get(category, 0) * BASE_UNIT_ML))
        if amount > 0:
            ingredients.append({"id": ing["id"], "amount": amount})

    return {"glass": glass, "ingredients": ingredients}


def enrich(cocktail, ing_by_id, glasses):
    """Resolve names/images and compute totals for display."""
    total_amount = 0.0
    alcohol = 0.0
    rows = []
    for item in cocktail["ingredients"]:
        info = ing_by_id.get(item["id"], {})
        rows.append({
            "id": item["id"],
            "name": info.get("name", item["id"]),
            "image": info.get("image", ""),
            "amount": item["amount"],
            "unit": info.get("unit", "ml"),
        })
        total_amount += item["amount"]
        alcohol += item["amount"] * info.get("alc", 0) / 100

    glass = next((g for g in glasses if g["id"] == cocktail["glass"]), None)
    return {
        "name": cocktail.get("name", "Namenloser Sloptail"),
        "description": cocktail.get("description", ""),
        "class_name": cocktail.get("class_name", ""),
        "glass": cocktail["glass"],
        "glass_name": glass["name"] if glass else cocktail["glass"],
        "ingredients": rows,
        "total_amount": int(round(total_amount)),
        "alcohol_content": round(alcohol * 100 / total_amount, 1) if total_amount else 0.0,
    }
