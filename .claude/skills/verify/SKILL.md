---
name: verify
description: Build, launch and drive the CocktailMaster FastAPI app to observe a change working end-to-end.
---

# Verifying CocktailMaster

FastAPI + Jinja templates + JSON files as the database. No test suite — verification
means driving the running app.

## Launch

Use the in-repo venv (`numpy`, `httpx`, `uvicorn` live there; the system python
does not have them):

```bash
./.venv/Scripts/python.exe -m uvicorn main:app --port 8077
```

Pick a port that is **not 8000 or 8003** — the user usually has the real app on
:8000 and the hardware scale service (`dzd.py`) on :8003. Check first and never
kill those:

```powershell
Get-NetTCPConnection -LocalPort 8000,8003 -State Listen -ErrorAction SilentlyContinue
```

There is no `--reload`; **restart the server after every edit** or you will verify
stale code.

## Teardown

`pkill -f uvicorn` does **not** work here (Windows). Kill by listening port:

```powershell
Get-NetTCPConnection -LocalPort 8077 -State Listen | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }
```

## Gotchas

- `GET /mix/{id}` spawns `dzd.py`, which grabs the serial port (COM4). The page
  still renders without the hardware — the scale just reports offline.
- Writing pages (`/manage/create`, `/sloptails/save`, `/data/*/save`,
  `/ingredients/toggle`) mutate the tracked JSON files in place. Prefer
  read-only routes; `git diff recipes.json` if you suspect you wrote something.
  The user's own running app may also change these files mid-session.
- Console output is cp1252 — prefix `PYTHONIOENCODING=utf-8` or German
  umlauts raise `UnicodeEncodeError`.
- Bump `?v=N` on `style.css` in `templates/base.html` when CSS changes, or the
  browser serves the old sheet.

## Driving the recipe maths

The ratio solver (`sloptails.solve_amounts`) is the heart of both `/sloptails`
and `/guided`. `/guided/state` is stateless and needs no LLM, which makes it the
cheapest way to exercise the solver end-to-end (`/sloptails/generate` needs
Ollama on :11434):

```bash
curl -s -X POST http://127.0.0.1:8077/guided/state -H 'Content-Type: application/json' \
  -d '{"class":"highball","chosen":{"spirit":"gin","sour":"lemon_juice","filler":"sprite"},
       "extras":[],"notes":["citrus"],"alcohol_free":false,"name":"T"}'
```

Check the returned `balance.axes`: `achieved` should land on `target` for every
axis the class asks for. When changing solver code, verify **volumes**, not just
the residual — a self-consistent solve can still produce wrong millilitres (see
the comp-normalization note in `sloptails.comp_vector`).
