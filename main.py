import json
from fastapi import FastAPI, Request, HTTPException, Form, File, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool
from contextlib import asynccontextmanager
import uuid
import copy
import random
import httpx
import os
import sys
import subprocess
import threading
import sloptails
import guided

# --- Scale service management -------------------------------------------------
# The hardware scale runs as a standalone process (dzd.py) that owns the serial
# port (COM4) and serves weights on :8003. We (re)start it whenever a user opens
# a /mix page so it is always freshly initialized.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_scale_proc = None
_scale_lock = threading.Lock()

def restart_scale_service():
    """Cleanly (re)start the scale service.

    Terminates any previously spawned instance first and waits for it to exit,
    so the serial port and HTTP port are released before the new process opens
    them. Returns the new process PID.
    """
    global _scale_proc
    with _scale_lock:
        if _scale_proc and _scale_proc.poll() is None:
            _scale_proc.terminate()
            try:
                _scale_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                _scale_proc.kill()
                _scale_proc.wait()
        _scale_proc = subprocess.Popen([sys.executable, "dzd.py"], cwd=BASE_DIR)
        return _scale_proc.pid

def _stop_scale_service():
    """Don't leave an orphaned scale process when the web app stops."""
    with _scale_lock:
        if _scale_proc and _scale_proc.poll() is None:
            _scale_proc.terminate()

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    _stop_scale_service()

app = FastAPI(lifespan=lifespan)

# Mount static files and setup templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load recipes
def load_recipes():
    with open("recipes.json", "r", encoding="utf-8") as f:
        return json.load(f)
    
def load_ingredients():
    with open("ingredients.json", "r", encoding="utf-8") as f:
        return json.load(f)

def load_glasses():
    with open("glasses.json", "r", encoding="utf-8") as f:
        return json.load(f)

def load_classes():
    with open("cocktail_classes.json", "r", encoding="utf-8") as f:
        return json.load(f)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    recipes = load_recipes()
    ingredients = load_ingredients()
    glasses = load_glasses()
    
    for recipe in recipes:
        alcohol_content = 0.0
        total_amount = 0.0
        available = True
        recipe["glass_info"] = next((g for g in glasses if g["id"] == recipe["glass"]), None)
        for ingredient in recipe["ingredients"]:
            ingredient_info = next((i for i in ingredients if i["id"] == ingredient["id"]), None)
            if ingredient_info:
                alcohol_content += (ingredient["amount"] * ingredient_info["alc"] / 100)
                total_amount += ingredient["amount"]
                if not ingredient_info.get("available", True):
                    available = False
        recipe["alcohol_content"] = round(alcohol_content*100/total_amount, 1) if total_amount else 0.0
        recipe["total_amount"] = total_amount
        recipe["alcfree"] = alcohol_content < 0.0001
        recipe["available"] = available

    # Move recipes with an out-of-stock ingredient to the end (stable sort).
    recipes.sort(key=lambda r: not r["available"])

    return templates.TemplateResponse(request, "index.html", {"recipes": recipes})

@app.get("/mix/{recipe_id}", response_class=HTMLResponse)
async def mix(request: Request, recipe_id: str):
    recipes = load_recipes()
    ingredients = load_ingredients()
    pending = _pending_sloptails.get(recipe_id)
    source = pending if pending else next((r for r in recipes if r["id"] == recipe_id), None)
    if not source:
        raise HTTPException(status_code=404, detail="Recipe not found")

    # Work on a copy so we never mutate a stored (pending) sloptail.
    recipe = copy.deepcopy(source)

    # Freshly (re)start the scale service so it is cleanly initialized for this mix.
    # Run off the event loop since terminating the old process can block briefly.
    await run_in_threadpool(restart_scale_service)

    computed_steps = []
    for ingredient in recipe["ingredients"]:
        ingredient_info = next((i for i in ingredients if i["id"] == ingredient["id"]), None)
        computed_steps.append(
            {
                "name": ingredient_info["name"] if ingredient_info else "Unknown",
                "image": ingredient_info["image"] if ingredient_info else "",
                "amount": ingredient["amount"],
                "unit": ingredient_info["unit"] if ingredient_info and "unit" in ingredient_info else "ml"
            })
    recipe["ingredients"] = computed_steps
    recipe["glass_info"] = next((g for g in load_glasses() if g["id"] == recipe["glass"]), None)
    # Only unsaved sloptails can be saved from the finish screen.
    recipe["savable"] = bool(pending)
    recipe["pending_id"] = recipe_id if pending else None

    return templates.TemplateResponse(request, "mix.html", {"recipe": recipe})

# Proxy endpoint to talk to your hardware scale safely
@app.get("/api/scale")
async def get_scale_weight():
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get("http://localhost:8003/weight")
            return response.json()
        except httpx.RequestError:
            return {"grams": 0.0, "stable": False, "error": "Scale offline"}


# Route to view recipe creator
@app.get("/manage", response_class=HTMLResponse)
async def manage_drinks(request: Request):
    ingredients = load_ingredients()
    return templates.TemplateResponse(request, "manage.html", {"ingredients": ingredients})

# --- Master data editor (structured forms) -----------------------------------
def _write_json(fname, data):
    with open(os.path.join(BASE_DIR, fname), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def _num(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

def _clean_ingredient(obj):
    """Build a tidy ingredient dict from posted form data (omitting defaults)."""
    ing = {
        "id": str(obj.get("id", "")).strip(),
        "name": str(obj.get("name", "")).strip(),
        "image": str(obj.get("image", "")).strip(),
        "alc": round(_num(obj.get("alc")), 2),
    }
    cat = str(obj.get("cat", "")).strip()
    if cat:
        ing["cat"] = cat
    comp = {k: _num(v) for k, v in (obj.get("comp") or {}).items() if _num(v) != 0}
    if comp:
        ing["comp"] = comp
    taste = [str(t).strip() for t in (obj.get("taste") or []) if str(t).strip()]
    if taste:
        ing["taste"] = taste
    unit = str(obj.get("unit", "")).strip()
    if unit and unit != "ml":
        ing["unit"] = unit
    gpp = _num(obj.get("gram_per_piece"))
    if gpp > 0:
        ing["gram_per_piece"] = int(gpp) if gpp == int(gpp) else gpp
    if obj.get("fillable"):
        ing["fillable"] = True
    if obj.get("available") is False:
        ing["available"] = False
    return ing

def _clean_glass(obj):
    glass = {"id": str(obj.get("id", "")).strip(), "name": str(obj.get("name", "")).strip()}
    image = str(obj.get("image", "")).strip()
    if image:
        glass["image"] = image
    glass["volume"] = int(_num(obj.get("volume")))
    glass["ice"] = int(_num(obj.get("ice")))
    return glass

@app.get("/data", response_class=HTMLResponse)
async def data_editor(request: Request):
    return templates.TemplateResponse(request, "data.html", {
        "ingredients": load_ingredients(),
        "glasses": load_glasses(),
        "classes": load_classes(),
        "categories": sloptails.CATEGORIES,
        "taste_notes": sloptails.TASTE_NOTES,
    })

@app.post("/data/upload")
async def data_upload(image: UploadFile = File(...)):
    if not image.filename:
        return {"ok": False, "error": "Keine Datei."}
    ext = os.path.splitext(image.filename)[1].lower()
    if ext not in (".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg"):
        return {"ok": False, "error": "Dateityp nicht erlaubt."}
    name = f"{uuid.uuid4().hex}{ext}"
    dest = os.path.join(BASE_DIR, "static", "images", "ingredients", name)
    with open(dest, "wb") as f:
        f.write(await image.read())
    return {"ok": True, "url": f"/static/images/ingredients/{name}"}

@app.post("/data/ingredients/save")
async def data_ingredient_save(request: Request):
    ing = _clean_ingredient(await request.json())
    if not ing["id"]:
        return {"ok": False, "error": "ID fehlt."}
    data = load_ingredients()
    for idx, existing in enumerate(data):
        if existing.get("id") == ing["id"]:
            data[idx] = ing
            break
    else:
        data.append(ing)
    _write_json("ingredients.json", data)
    return {"ok": True}

@app.post("/data/ingredients/delete")
async def data_ingredient_delete(request: Request):
    iid = (await request.json()).get("id")
    _write_json("ingredients.json", [e for e in load_ingredients() if e.get("id") != iid])
    return {"ok": True}

@app.post("/data/glasses/save")
async def data_glass_save(request: Request):
    glass = _clean_glass(await request.json())
    if not glass["id"]:
        return {"ok": False, "error": "ID fehlt."}
    data = load_glasses()
    for idx, existing in enumerate(data):
        if existing.get("id") == glass["id"]:
            data[idx] = glass
            break
    else:
        data.append(glass)
    _write_json("glasses.json", data)
    return {"ok": True}

@app.post("/data/glasses/delete")
async def data_glass_delete(request: Request):
    gid = (await request.json()).get("id")
    _write_json("glasses.json", [e for e in load_glasses() if e.get("id") != gid])
    return {"ok": True}

@app.post("/data/classes/save")
async def data_class_save(request: Request):
    obj = await request.json()
    key = str(obj.get("key", "")).strip()
    if not key:
        return {"ok": False, "error": "Key fehlt."}
    ratios = {k: _num(v) for k, v in (obj.get("ratios") or {}).items() if _num(v) != 0}
    data = load_classes()
    data[key] = {
        "name": str(obj.get("name", "")).strip(),
        "description": str(obj.get("description", "")).strip(),
        "glass": str(obj.get("glass", "")).strip() or "Longdrink",
        "ratios": ratios,
    }
    _write_json("cocktail_classes.json", data)
    return {"ok": True}

@app.post("/data/classes/delete")
async def data_class_delete(request: Request):
    key = (await request.json()).get("key")
    data = load_classes()
    data.pop(key, None)
    _write_json("cocktail_classes.json", data)
    return {"ok": True}

# Printable overview of all configured ingredients and recipes.
@app.get("/overview", response_class=HTMLResponse)
async def overview(request: Request):
    recipes = load_recipes()
    ingredients = load_ingredients()
    glasses = load_glasses()
    ing_by_id = {i["id"]: i for i in ingredients}

    for recipe in recipes:
        recipe["glass_info"] = next((g for g in glasses if g["id"] == recipe["glass"]), None)
        total_amount = 0.0
        alcohol_content = 0.0
        rows = []
        for ingredient in recipe["ingredients"]:
            info = ing_by_id.get(ingredient["id"])
            rows.append({
                "name": info["name"] if info else ingredient["id"],
                "amount": ingredient["amount"],
                "unit": info.get("unit", "ml") if info else "ml",
            })
            if info:
                total_amount += ingredient["amount"]
                alcohol_content += ingredient["amount"] * info["alc"] / 100
        recipe["rows"] = rows
        recipe["total_amount"] = total_amount
        recipe["alcohol_content"] = round(alcohol_content * 100 / total_amount, 1) if total_amount else 0.0

    return templates.TemplateResponse(
        request, "overview.html", {"recipes": recipes, "ingredients": ingredients}
    )

# --- Sloptails: AI-generated cocktails (multi-step, rail-guided) --------------
@app.get("/sloptails", response_class=HTMLResponse)
async def sloptails_page(request: Request):
    return templates.TemplateResponse(
        request, "sloptails.html", {
            "taste_notes": sloptails.taste_note_options(),
            "classes": load_classes(),
        }
    )

async def _ollama_json(client, messages, temperature=0.8, num_predict=None):
    """One constrained JSON turn against the local model.

    `num_predict` caps how much the model may generate: for a short answer that
    is the difference between a snappy reply and waiting on a model that decided
    to write an essay.
    """
    options = {"temperature": temperature, "top_p": 0.95}
    if num_predict:
        options["num_predict"] = num_predict
    payload = {
        "model": sloptails.OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "format": "json",
        "options": options,
        "keep_alive": sloptails.OLLAMA_KEEP_ALIVE,
    }
    resp = await client.post(f"{sloptails.OLLAMA_URL}/api/chat", json=payload)
    resp.raise_for_status()
    return json.loads(resp.json().get("message", {}).get("content", ""))

@app.post("/sloptails/generate")
async def sloptails_generate(
    taste_notes: list = Form(default=[]),
    alcohol_free: bool = Form(default=False),
    cocktail_class: str = Form(default=""),
):
    ingredients = load_ingredients()
    glasses = load_glasses()
    classes = load_classes()
    ing_by_id = {i["id"]: i for i in ingredients}
    notes = [n for n in taste_notes if isinstance(n, str) and n.strip()]

    # 1) pick a cocktail class: an explicit choice (for testing) or random
    if cocktail_class in classes:
        class_key, class_def = cocktail_class, classes[cocktail_class]
    else:
        class_key, class_def = random.choice(list(classes.items()))

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            # 2) pick one ingredient per role, shown only that category's options
            chosen = {}
            for category in class_def.get("ratios", {}):
                candidates = sloptails.candidates_for(category, ingredients, alcohol_free)
                if not candidates:
                    continue
                random.shuffle(candidates)  # avoid position bias; resolve against this order
                raw = await _ollama_json(
                    client, sloptails.build_ingredient_messages(
                        random.sample(notes, len(notes)), category, candidates), 1.5)
                cidx = sloptails.parse_choice(raw, "ingredient", len(candidates)) or 1
                chosen[category] = candidates[cidx - 1]

            if not chosen:
                return {"ok": False, "error": "Keine verfügbaren Zutaten für diese Rezeptklasse."}

            # 3) solve amounts as a linear combination of ingredient comp vectors
            ratios = class_def.get("ratios", {})
            columns, seen = [], set()
            for ing in chosen.values():
                if ing["id"] not in seen:
                    columns.append(ing)
                    seen.add(ing["id"])

            amounts, rel = sloptails.solve_amounts(columns, ratios)

            # Overconstrained -> add a pure ingredient for each non-pure role so the
            # target becomes reachable; reprompt the model (pure options only) when
            # several pures exist for that role.
            if rel > sloptails.OVERCONSTRAINED_REL:
                for category, ing in chosen.items():
                    if sloptails.is_pure(ing, category):
                        continue
                    pures = [i for i in sloptails.candidates_for(category, ingredients, alcohol_free)
                             if sloptails.is_pure(i, category) and i["id"] not in seen]
                    if not pures:
                        continue
                    if len(pures) == 1:
                        pick = pures[0]
                    else:
                        random.shuffle(pures)  # avoid position bias; resolve against this order
                        raw = await _ollama_json(
                            client, sloptails.build_ingredient_messages(
                                random.sample(notes, len(notes)), category, pures), 1.5)
                        pidx = sloptails.parse_choice(raw, "ingredient", len(pures)) or 1
                        pick = pures[pidx - 1]
                    columns.append(pick)
                    seen.add(pick["id"])
                amounts, rel = sloptails.solve_amounts(columns, ratios)

            cocktail = sloptails.build_recipe(columns, amounts, class_def, glasses)
            cocktail["class_name"] = class_def.get("name", class_key)

            # 4) creative name + description (graceful fallback)
            chosen_names = [ing_by_id[i["id"]]["name"] for i in cocktail["ingredients"]
                            if i["id"] != "ice" and i["id"] in ing_by_id]
            try:
                raw = await _ollama_json(
                    client, sloptails.build_name_messages(
                        class_def.get("name", ""), chosen_names, random.sample(notes, len(notes))), 1.5)
                cocktail["name"] = str(raw.get("name") or "").strip() or class_def.get("name", "Sloptail")
                cocktail["description"] = str(raw.get("description") or "").strip()
            except (httpx.HTTPStatusError, ValueError, KeyError):
                base = chosen_names[0] if chosen_names else class_def.get("name", "Sloptail")
                cocktail["name"] = f"{base} {class_def.get('name', 'Sloptail')}"
                cocktail["description"] = class_def.get("description", "")
    except (httpx.RequestError, httpx.HTTPStatusError):
        return {"ok": False, "offline": True,
                "error": f"KI nicht erreichbar. Läuft Ollama ({sloptails.OLLAMA_MODEL}) unter {sloptails.OLLAMA_URL}?"}
    except (json.JSONDecodeError, ValueError, KeyError):
        return {"ok": False, "error": "Die KI-Antwort konnte nicht gelesen werden."}

    return {"ok": True, "source": "ollama", "cocktail": sloptails.enrich(cocktail, ing_by_id, glasses)}

# Generated sloptails are not persisted until the user saves them; we keep them
# in memory just long enough to mix (and optionally save) them.
_pending_sloptails = {}

def _normalize_sloptail(payload):
    """Turn a client-provided cocktail into a validated recipe dict, or None."""
    if not isinstance(payload, dict):
        return None
    ingredients_pool = load_ingredients()
    ing_by_id = {i["id"]: i for i in ingredients_pool}
    glasses = load_glasses()

    ingredients = []
    for item in payload.get("ingredients", []):
        if not isinstance(item, dict):
            continue
        iid = item.get("id")
        if iid not in ing_by_id:
            continue
        try:
            amount = int(item.get("amount", 0))
        except (TypeError, ValueError):
            continue
        if amount > 0:
            ingredients.append({"id": iid, "amount": amount})
    if not ingredients:
        return None

    glass = payload.get("glass")
    if not any(g["id"] == glass for g in glasses):
        glass = glasses[0]["id"] if glasses else "Longdrink"

    return {
        "name": str(payload.get("name") or "Sloptail").strip()[:80],
        "glass": glass,
        # No cover image for AI drinks -> reuse the first ingredient's image.
        "image": ing_by_id[ingredients[0]["id"]].get("image", "/static/images/default-cocktail.jpg"),
        "ingredients": ingredients,
    }

@app.post("/sloptails/mix")
async def sloptails_mix(request: Request):
    recipe = _normalize_sloptail(await request.json())
    if not recipe:
        raise HTTPException(status_code=400, detail="Ungültiger Cocktail.")
    sid = "slop-" + uuid.uuid4().hex[:8]
    _pending_sloptails[sid] = recipe
    return {"id": sid}

@app.post("/sloptails/save")
async def sloptails_save(pending_id: str = Form(...)):
    recipe = _pending_sloptails.get(pending_id)
    if not recipe:
        raise HTTPException(status_code=404, detail="Sloptail nicht gefunden.")
    recipes = load_recipes()
    new_recipe = {
        "id": str(uuid.uuid4())[:8],
        "name": recipe["name"],
        "glass": recipe["glass"],
        "image": recipe["image"],
        "ingredients": recipe["ingredients"],
    }
    recipes.append(new_recipe)
    with open("recipes.json", "w", encoding="utf-8") as f:
        json.dump(recipes, f, indent=2, ensure_ascii=False)
    _pending_sloptails.pop(pending_id, None)
    return {"ok": True, "id": new_recipe["id"]}

# --- Guided creation: the user picks, the solver keeps the drink in balance ---
@app.get("/guided", response_class=HTMLResponse)
async def guided_page(request: Request):
    return templates.TemplateResponse(
        request, "guided.html", {
            "taste_notes": sloptails.taste_note_options(),
            "classes": load_classes(),
        }
    )

@app.post("/guided/state")
async def guided_state(request: Request):
    """Re-derive the whole page from the current selection.

    Stateless: the client owns the picks and posts them back on every change, so
    a reload or a second tab can never disagree with the drink on screen.
    """
    payload = await request.json()
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Ungültige Anfrage.")

    classes = load_classes()
    class_def = classes.get(str(payload.get("class") or ""))
    if not class_def:
        raise HTTPException(status_code=404, detail="Unbekannte Rezeptklasse.")

    chosen = payload.get("chosen")
    extras = payload.get("extras")
    notes = payload.get("notes")
    state = await run_in_threadpool(
        guided.build_state,
        class_def=class_def,
        ingredients=load_ingredients(),
        glasses=load_glasses(),
        chosen_ids=chosen if isinstance(chosen, dict) else {},
        extra_ids=extras if isinstance(extras, list) else [],
        notes=notes if isinstance(notes, list) else [],
        alcohol_free=bool(payload.get("alcohol_free")),
        name=str(payload.get("name") or ""),
    )
    return {"ok": True, **state}

@app.post("/guided/warmup")
async def guided_warmup():
    """Pull the model into memory when the page opens.

    The first turn against a cold model pays the load cost, which is exactly the
    turn the user is waiting on. Fire this on page load and the naming that
    follows is warm. Fire-and-forget: the caller ignores the outcome.
    """
    try:
        async with httpx.AsyncClient(timeout=guided.WARMUP_TIMEOUT) as client:
            await client.post(
                f"{sloptails.OLLAMA_URL}/api/chat",
                json={"model": sloptails.OLLAMA_MODEL, "messages": [],
                      "keep_alive": sloptails.OLLAMA_KEEP_ALIVE},
            )
    except (httpx.RequestError, httpx.HTTPStatusError):
        return {"ok": False, "offline": True}
    return {"ok": True}

@app.post("/guided/name")
async def guided_name(request: Request):
    """Invent a name for the drink as it stands.

    Deliberately its own endpoint: /guided/state stays instant and works with
    the model offline, and this can be superseded (the client aborts it) every
    time the user picks something else. Any failure returns ok=False rather than
    an error -- a nameless drink is still a drink.
    """
    payload = await request.json()
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Ungültige Anfrage.")

    names = [str(n).strip()[:40] for n in (payload.get("ingredients") or [])
             if str(n).strip()][:6]
    if not names:
        return {"ok": False}
    notes = [str(n).strip()[:24] for n in (payload.get("notes") or []) if str(n).strip()][:4]

    messages = guided.build_name_messages(
        str(payload.get("class_name") or "").strip()[:40], names, notes)
    try:
        async with httpx.AsyncClient(timeout=guided.NAME_TIMEOUT) as client:
            raw = await _ollama_json(client, messages,
                                     temperature=guided.NAME_TEMPERATURE,
                                     num_predict=guided.NAME_MAX_TOKENS)
    except (httpx.RequestError, httpx.HTTPStatusError, httpx.TimeoutException):
        return {"ok": False, "offline": True}
    except (json.JSONDecodeError, ValueError, KeyError):
        return {"ok": False}

    name = guided.clean_name(raw.get("name") if isinstance(raw, dict) else "")
    return {"ok": bool(name), "name": name}

# Ingredient stock management: toggle availability when something runs out.
@app.get("/ingredients", response_class=HTMLResponse)
async def manage_ingredients(request: Request):
    ingredients = load_ingredients()
    for ing in ingredients:
        ing.setdefault("available", True)
    return templates.TemplateResponse(request, "ingredients.html", {"ingredients": ingredients})

@app.post("/ingredients/toggle")
async def toggle_ingredient(ingredient_id: str = Form(...), available: bool = Form(...)):
    ingredients = load_ingredients()
    target = next((i for i in ingredients if i["id"] == ingredient_id), None)
    if not target:
        raise HTTPException(status_code=404, detail="Ingredient not found")
    target["available"] = available
    with open("ingredients.json", "w", encoding="utf-8") as f:
        json.dump(ingredients, f, indent=2, ensure_ascii=False)
    return {"ok": True, "id": ingredient_id, "available": available}

# Process form data from creator UI
@app.post("/manage/create")
async def create_drink(
    request: Request,
    name: str = Form(...),
    glass: str = Form(...), # This will match the glass 'id' from our new select menu
    ingredient_id: list = Form(...),
    ingredient_amount: list = Form(...),
    image_file: UploadFile = File(...)
    ):
    

    glasses_pool = load_glasses()
    matched_glass = next((g for g in glasses_pool if g["id"] == glass), None)
    
    if not matched_glass:
        raise HTTPException(status_code=400, detail="Selected glass type is invalid.")
    
    # Calculate the total weight/volume of all added items
    total_volume = sum(int(amount) for amount in ingredient_amount if amount)
    glass_max = matched_glass["volume"]
    
    if total_volume > glass_max:
        # Throw an error that tells the user exactly how much they overshot by
        raise HTTPException(
            status_code=400, 
            detail=f"Recipe exceeds glass capacity! Total ingredients: {total_volume}ml. '{matched_glass['name']}' max capacity: {glass_max}ml."
        )

    # 2. File handling (Keep your existing unique filename logic here)
    image_path = "/static/images/default-cocktail.jpg"
    if image_file and image_file.filename:
        file_extension = os.path.splitext(image_file.filename)[1]
        unique_filename = f"{uuid.uuid4().hex}{file_extension}"
        file_location = os.path.join("static/images", unique_filename)
        with open(file_location, "wb") as buffer:
            content = await image_file.read()
            buffer.write(content)
        image_path = f"/static/images/{unique_filename}"
    
    ingredients_pool = load_ingredients()
    compiled_ingredients = []

    
    for i in range(len(ingredient_id)):
        if not ingredient_id[i]: continue # Skip blanks
        
        # Match chosen ID against ingredient database to pull names/images automatically
        compiled_ingredients.append({
            "id": ingredient_id[i],
            "amount": int(ingredient_amount[i])
        })
            
    if not name or not compiled_ingredients:
        raise HTTPException(status_code=400, detail="Missing required parameters")

    # Read, append, and rewrite recipes file safely
    recipes = load_recipes()
    new_recipe = {
        "id": str(uuid.uuid4())[:8], # Creates a short unique id string
        "name": name,
        "glass": glass,
        "image": image_path, # Default placeholder image
        "ingredients": compiled_ingredients
    }
    
    recipes.append(new_recipe)
    with open("recipes.json", "w", encoding="utf-8") as f:
        json.dump(recipes, f, indent=2, ensure_ascii=False)
        
    return RedirectResponse(url="/", status_code=303)