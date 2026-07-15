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
import numpy as np

# Configurable so nothing is hard-coded when the model changes.
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")

# Categories an ingredient can serve, in display / build order.
CATEGORIES = ["spirit", "sweet", "sour", "bitter", "filler"]

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


# --- Step 3: solve amounts as a linear combination of comp vectors -----------
# The class `ratios` are the target vector over the axes (CATEGORIES); each
# ingredient's `comp` is its vector. We seek non-negative amounts x with
# sum_i x_i * comp_i ~= ratios. Ridge regularization makes underconstrained
# systems fall back to an even (minimum-norm) split; a large residual flags an
# overconstrained system that needs pure ingredients added (handled in main.py).
SOLVE_RIDGE = 1e-3
OVERCONSTRAINED_REL = 0.05  # relative residual above which we add pure ingredients


def comp_vector(ingredient, axes=None):
    """An ingredient's composition as fractions of itself (the parts sum to 1).

    Only the *proportions* in `comp` were ever meant to carry meaning -- cola's
    {sweet 0.25, filler 0.75} and sprite's {sweet 1, filler 3} say the same
    thing on different scales. Normalizing makes that explicit, and it is what
    lets build_recipe spend the solved x as millilitres: one unit of x is one
    unit of *volume*, whose character is spread over the axes by these
    fractions. Without it a raw comp summing to 4 would quietly act as a 4x
    concentrate and eat only a quarter of the volume it was solved for.
    """
    comp = ingredient.get("comp", {}) or {}
    vec = [float(comp.get(ax, 0.0)) for ax in (axes or CATEGORIES)]
    total = sum(vec)
    return [v / total for v in vec] if total > 0 else vec


def is_pure(ingredient, axis):
    """True if the ingredient's composition touches only `axis`."""
    comp = ingredient.get("comp", {}) or {}
    return [k for k, v in comp.items() if float(v) > 0] == [axis]


def _nnls(A, b, tol=1e-10):
    """Non-negative least squares (Lawson-Hanson active-set)."""
    A = np.asarray(A, float)
    b = np.asarray(b, float)
    n = A.shape[1]
    x = np.zeros(n)
    passive = np.zeros(n, dtype=bool)
    w = A.T @ (b - A @ x)
    for _ in range(3 * n + 5):
        if passive.all() or w[~passive].max(initial=-np.inf) <= tol:
            break
        active = np.where(~passive)[0]
        passive[active[np.argmax(w[active])]] = True
        while True:
            idx = np.where(passive)[0]
            sp = np.linalg.lstsq(A[:, idx], b, rcond=None)[0]
            if (sp > tol).all():
                x[idx] = sp
                break
            neg = sp <= tol
            alpha = (x[idx][neg] / (x[idx][neg] - sp[neg])).min()
            x[idx] = x[idx] + alpha * (sp - x[idx])
            drop = idx[x[idx] <= tol]
            passive[drop] = False
            x[drop] = 0.0
        w = A.T @ (b - A @ x)
    return x


def solve_amounts(columns, ratios, axes=None, ridge=SOLVE_RIDGE):
    """Non-negative ratio-space amounts, one per column. Returns (x, rel_residual)."""
    axes = axes or CATEGORIES
    n = len(columns)
    if n == 0:
        return np.zeros(0), 0.0
    A = np.array([comp_vector(c, axes) for c in columns], float).T  # (axes, cols)
    b = np.array([float(ratios.get(ax, 0.0)) for ax in axes], float)
    A_aug = np.vstack([A, np.sqrt(ridge) * np.eye(n)])
    b_aug = np.concatenate([b, np.zeros(n)])
    x = _nnls(A_aug, b_aug)
    residual = float(np.linalg.norm(A @ x - b))
    return x, residual / (float(np.linalg.norm(b)) or 1.0)


def build_recipe(columns, amounts, class_def, glasses):
    """Scale ratio-space amounts to millilitres (fills glass minus ice)."""
    glass_id = class_def.get("glass", "Longdrink")
    glass = next((g for g in (glasses or []) if g.get("id") == glass_id), {})
    ice_g = int(glass.get("ice", 0))
    liquid = max(0.0, float(glass.get("volume", 0)) - ice_g)

    ingredients = []
    if ice_g > 0:
        ingredients.append({"id": "ice", "amount": ice_g})

    amounts = list(amounts)
    total = float(sum(amounts))
    if total <= 0 and columns:  # degenerate solution -> even split
        amounts = [1.0] * len(columns)
        total = float(len(columns))

    for col, amount in zip(columns, amounts):
        ml = int(round(float(amount) / total * liquid)) if total > 0 else 0
        if ml > 0:
            ingredients.append({"id": col["id"], "amount": ml})

    return {"glass": glass_id, "ingredients": ingredients}


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
