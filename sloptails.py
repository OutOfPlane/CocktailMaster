"""Harness for AI-generated 'Sloptails'.

Builds the prompt for a local Ollama model (phi-4-mini), validates the model's
JSON answer against the known ingredients/glasses and enriches it for display.
The actual HTTP call lives in main.py (async httpx); everything model-agnostic
and testable lives here.
"""
import os

# Configurable so nothing is hard-coded when the model changes.
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "phi4-mini")

# Selectable flavor profiles shown in the UI.
FLAVORS = ["Fruchtig", "Würzig", "Komplex", "Süß", "Herb", "Erfrischend", "Cremig"]


def build_messages(selected_ingredients, flavor, glasses):
    """Build the chat messages for Ollama.

    selected_ingredients: list of ingredient dicts (id, name, alc). Ingredients
    are presented numbered (1..N); the model references them by that number,
    which small models handle far more reliably than echoing string ids.
    """
    ing_lines = "\n".join(
        f'{pos + 1}. {i["name"]} ({i.get("alc", 0)}% Alk.)'
        for pos, i in enumerate(selected_ingredients)
    )
    glass_lines = "\n".join(
        f'- {g["id"]}: {g["name"]} ({g.get("volume", "?")} ml)' for g in glasses
    )

    system = (
        "Du bist ein kreativer Barkeeper, der ausgefallene 'Sloptails' erfindet. "
        "Du antwortest ausschließlich mit gültigem JSON, niemals mit Fließtext."
    )
    user = f"""Erfinde einen Cocktail mit dem Geschmacksprofil "{flavor}".
Wähle aus diesen nummerierten Zutaten:
{ing_lines}

Verfügbare Gläser:
{glass_lines}

Antworte mit JSON in genau diesem Format:
{{
  "name": "<kreativer deutscher Name>",
  "glass": "<glas-id aus der Liste>",
  "description": "<ein kurzer Satz>",
  "ingredients": [{{"id": <ZUTAT-NUMMER>, "amount": <menge in ml als Zahl>}}]
}}

Regeln:
- "id" ist die NUMMER der Zutat aus der obigen Liste (z.B. 1, 2, 3).
- Wähle 2 bis 5 Zutaten mit sinnvollen Mengen (ca. 20-150 ml je Zutat).
- Die Gesamtmenge darf das gewählte Glasvolumen nicht überschreiten.
- Treffe das Geschmacksprofil "{flavor}"."""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _resolve_ingredient_id(item, selected):
    """Map whatever the model put in an ingredient entry back to a real id.

    Accepts (in order): the 1-based list number, the exact string id, or the
    ingredient name. Returns the resolved id or None.
    """
    allowed = {i["id"] for i in selected}
    by_index = {str(pos + 1): i["id"] for pos, i in enumerate(selected)}
    by_name = {i["name"].strip().lower(): i["id"] for i in selected}

    ref = item.get("id")
    if isinstance(ref, bool):
        ref = None
    if isinstance(ref, (int, float)):
        return by_index.get(str(int(ref)))
    if isinstance(ref, str):
        s = ref.strip()
        if s in allowed:
            return s
        if s in by_index:
            return by_index[s]
        if s.lower() in by_name:
            return by_name[s.lower()]

    name = str(item.get("name", "")).strip().lower()
    return by_name.get(name)


def validate_cocktail(raw, selected, glasses):
    """Validate/normalize the model's JSON. Returns (cocktail, error).

    selected: ordered list of the ingredient dicts the user picked.
    """
    if not isinstance(raw, dict):
        return None, "Die KI-Antwort hatte kein gültiges Format."

    name = str(raw.get("name") or "Namenloser Sloptail").strip()
    description = str(raw.get("description") or "").strip()

    glass = raw.get("glass")
    if not any(g["id"] == glass for g in glasses):
        glass = glasses[0]["id"] if glasses else "Longdrink"

    ingredients = []
    seen = set()
    for item in raw.get("ingredients", []):
        if not isinstance(item, dict):
            continue
        iid = _resolve_ingredient_id(item, selected)
        if iid is None or iid in seen:
            continue
        try:
            amount = int(round(float(item.get("amount", 0))))
        except (TypeError, ValueError):
            continue
        if amount <= 0:
            continue
        ingredients.append({"id": iid, "amount": amount})
        seen.add(iid)

    if not ingredients:
        return None, "Die KI hat keine gültigen Zutaten geliefert."

    return {"name": name, "glass": glass, "description": description, "ingredients": ingredients}, None


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
        })
        total_amount += item["amount"]
        alcohol += item["amount"] * info.get("alc", 0) / 100

    glass = next((g for g in glasses if g["id"] == cocktail["glass"]), None)
    return {
        "name": cocktail["name"],
        "description": cocktail["description"],
        "glass": cocktail["glass"],
        "glass_name": glass["name"] if glass else cocktail["glass"],
        "ingredients": rows,
        "total_amount": int(round(total_amount)),
        "alcohol_content": round(alcohol * 100 / total_amount, 1) if total_amount else 0.0,
    }
