import json
from fastapi import FastAPI, Request, HTTPException, Form, File, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool
from contextlib import asynccontextmanager
import uuid
import httpx
import os
import sys
import subprocess
import threading

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

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    recipes = load_recipes()
    ingredients = load_ingredients()
    glasses = load_glasses()
    
    for recipe in recipes:
        alcohol_content = 0.0
        total_amount = 0.0
        recipe["glass_info"] = next((g for g in glasses if g["id"] == recipe["glass"]), None)
        for ingredient in recipe["ingredients"]:
            ingredient_info = next((i for i in ingredients if i["id"] == ingredient["id"]), None)
            if ingredient_info:
                alcohol_content += (ingredient["amount"] * ingredient_info["alc"] / 100)
                total_amount += ingredient["amount"]
        recipe["alcohol_content"] = round(alcohol_content*100/total_amount, 1)
        recipe["total_amount"] = total_amount
        recipe["alcfree"] = alcohol_content < 0.0001


    return templates.TemplateResponse(request, "index.html", {"recipes": recipes})

@app.get("/mix/{recipe_id}", response_class=HTMLResponse)
async def mix(request: Request, recipe_id: str):
    recipes = load_recipes()
    ingredients = load_ingredients()
    recipe = next((r for r in recipes if r["id"] == recipe_id), None)
    if not recipe:
        raise HTTPException(status_code=404, detail="Recipe not found")

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
                "amount": ingredient["amount"]
            })
    recipe["ingredients"] = computed_steps
    
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