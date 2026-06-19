import json
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
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