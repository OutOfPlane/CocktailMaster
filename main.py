import json
from fastapi import FastAPI, Request, HTTPException, Form, File, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uuid
import httpx

app = FastAPI()

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

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    recipes = load_recipes()
    return templates.TemplateResponse(request, "index.html", {"recipes": recipes})

@app.get("/mix/{recipe_id}", response_class=HTMLResponse)
async def mix(request: Request, recipe_id: str):
    recipes = load_recipes()
    ingredients = load_ingredients()
    recipe = next((r for r in recipes if r["id"] == recipe_id), None)
    if not recipe:
        raise HTTPException(status_code=404, detail="Recipe not found")
    
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
async def create_drink(request: Request):
    form_data = await request.form()
    
    # Extract structural fields
    drink_name = form_data.get("name")
    glass_type = form_data.get("glass")
    
    # Process dynamically generated arrays of ingredients
    ing_ids = form_data.getlist("ingredient_id")
    ing_amounts = form_data.getlist("ingredient_amount")
    
    ingredients_pool = load_ingredients()
    compiled_ingredients = []
    
    for i in range(len(ing_ids)):
        if not ing_ids[i]: continue # Skip blanks
        
        # Match chosen ID against ingredient database to pull names/images automatically
        compiled_ingredients.append({
            "id": ing_ids[i],
            "amount": int(ing_amounts[i])
        })
            
    if not drink_name or not compiled_ingredients:
        raise HTTPException(status_code=400, detail="Missing required parameters")

    # Read, append, and rewrite recipes file safely
    recipes = load_recipes()
    new_recipe = {
        "id": str(uuid.uuid4())[:8], # Creates a short unique id string
        "name": drink_name,
        "glass": glass_type,
        "image": "/static/images/default-cocktail.jpg", # Default placeholder image
        "ingredients": compiled_ingredients
    }
    
    recipes.append(new_recipe)
    with open("recipes.json", "w", encoding="utf-8") as f:
        json.dump(recipes, f, indent=2, ensure_ascii=False)
        
    return RedirectResponse(url="/", status_code=303)