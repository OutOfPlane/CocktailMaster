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
    with open("recipes.json", "r") as f:
        return json.load(f)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    recipes = load_recipes()
    return templates.TemplateResponse(request, "index.html", {"recipes": recipes})

@app.get("/mix/{recipe_id}", response_class=HTMLResponse)
async def mix(request: Request, recipe_id: str):
    recipes = load_recipes()
    recipe = next((r for r in recipes if r["id"] == recipe_id), None)
    if not recipe:
        raise HTTPException(status_code=404, detail="Recipe not found")
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