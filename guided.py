"""Guided cocktail creation: the user picks, the maths keeps the drink in balance.

Same rails as the AI Sloptails -- the class ratios from cocktail_classes.json are
the target vector, each ingredient's `comp` is its contribution, and
sloptails.solve_amounts turns a set of ingredients into millilitres. The
difference is who chooses: here a human does, and every choice is scored and
shown before it is made.

Two signals drive a recommendation:
  * flavour  -- does it hit the notes you asked for, does it share notes with
                what is already in the glass, does it clash with any of it
  * fit      -- with this ingredient in the mix, how closely can the class
                ratios still be met (ingredients are rarely pure: Aperitivo is
                spirit *and* sweet *and* bitter, so picking it changes what
                every other role has to do)

Because `comp` vectors are re-solved after every pick, the balance panel can show
what the whole drink actually became -- including the axes an ingredient hits
that the class never asked for (tonic's bitter, cola's sweet, ...).

Pure logic only; the HTTP layer lives in main.py.
"""
import numpy as np

from sloptails import (
    CATEGORIES,
    OVERCONSTRAINED_REL,
    TASTE_NOTES_DE,
    build_recipe,
    candidates_for,
    comp_vector,
    enrich,
    is_pure,
    solve_amounts,
)

# The role each ratio axis plays in the glass, as shown in the UI.
ROLE_LABELS = {
    "spirit": "Basis", "sweet": "Süße", "sour": "Säure",
    "bitter": "Bitter", "filler": "Filler",
}

# Ingredient `taste` lists are free-form, so fold the common variants onto the
# canonical vocabulary -- otherwise Sprite's "lemons" would never match a wish
# for "citrus".
TASTE_SYNONYMS = {
    "lemon": "citrus", "lemons": "citrus", "orange": "citrus", "lime": "citrus",
    "herbs": "herbal", "spice": "warm spice", "spices": "warm spice",
    "ginger": "warm spice", "cola nut": "caramel", "vegetal": "grassy",
    "sparkling": "bright", "sharp carbonation": "bright", "sharp": "bright",
    "refreshing": "fresh",
}

# Note pairs that generally fight in a glass. Only a mild penalty: it should
# nudge the ranking, not forbid a combination someone deliberately wants.
CLASHES = [
    ("creamy", "sour"), ("creamy", "citrus"), ("creamy", "bitter"),
    ("chocolate", "citrus"), ("chocolate", "sour"),
    ("smoky", "floral"), ("smoky", "tropical"), ("smoky", "creamy"),
    ("agave", "chocolate"), ("agave", "vanilla"),
]

# Two matching notes is already a strong signal; more shouldn't dominate.
_NOTE_SATURATION = 2.0
_CLASH_PENALTY = 0.25
# A recommendation weighs taste a little heavier than reachability -- a
# perfectly balanced drink nobody wants to drink is the wrong answer.
_FLAVOR_WEIGHT = 0.55


# --- Naming, tuned for latency ------------------------------------------------
# The name is invented *while* the user is still picking, so this prompt is
# built for speed, not prose: a couple of dozen prompt tokens, a hard cap on the
# answer, and a short timeout. A name that arrives late is worse than no name --
# the drink is complete without one, so every failure here stays silent.
NAME_TIMEOUT = 8.0
NAME_MAX_TOKENS = 24
NAME_MAX_WORDS = 4
NAME_TEMPERATURE = 1.2
WARMUP_TIMEOUT = 30.0  # a cold model can take a while to load; nobody waits on it


def build_name_messages(class_name, ingredient_names, notes):
    """One short turn: ingredients in, two or three words out."""
    system = ('Name the cocktail. 2-3 words, English, evocative, no quotes. '
              'JSON only: {"name":"..."}')
    user = ", ".join(ingredient_names)
    if class_name:
        user = f"{class_name}: {user}"
    if notes:
        user += f" | {', '.join(notes)}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def clean_name(value):
    """Small models like to add quotes, trailing punctuation or a whole sentence."""
    name = " ".join(str(value or "").split()).strip(" \"'`.,;:!?-—")
    if not name:
        return ""
    return " ".join(name.split(" ")[:NAME_MAX_WORDS])[:60]


def norm_note(note):
    n = str(note).strip().lower()
    return TASTE_SYNONYMS.get(n, n)


def notes_of(ingredient):
    return {norm_note(t) for t in ingredient.get("taste", []) or [] if str(t).strip()}


def _label(note):
    return TASTE_NOTES_DE.get(note, note.capitalize())


def _card(ingredient):
    """The bits of an ingredient the UI needs to draw a choice."""
    return {
        "id": ingredient["id"],
        "name": ingredient.get("name", ingredient["id"]),
        "image": ingredient.get("image", ""),
        "alc": ingredient.get("alc", 0),
        "comp": {k: v for k, v in (ingredient.get("comp") or {}).items()},
        "taste": [_label(n) for n in sorted(notes_of(ingredient))],
    }


def _dedupe(columns):
    out, seen = [], set()
    for c in columns:
        if c["id"] not in seen:
            out.append(c)
            seen.add(c["id"])
    return out


def _fit(columns, ratios):
    """1.0 = the class ratios are exactly reachable with these ingredients."""
    if not columns:
        return 0.0
    _, rel = solve_amounts(columns, ratios)
    return max(0.0, 1.0 - rel)


def flavor_score(candidate, desired, context_notes, context_ingredients):
    """Taste affinity in [0, 1] plus the German reasons behind it."""
    cnotes = notes_of(candidate)
    reasons = []

    hits = sorted(cnotes & desired)
    want = min(1.0, len(hits) / _NOTE_SATURATION)
    if hits:
        reasons.append("trifft " + ", ".join(_label(h) for h in hits))

    shared = cnotes & context_notes
    bridge = min(1.0, len(shared) / _NOTE_SATURATION)
    if shared:
        partners = [i.get("name", i["id"]) for i in context_ingredients
                    if notes_of(i) & cnotes]
        if partners:
            reasons.append("harmoniert mit " + ", ".join(partners[:2]))

    context = desired | context_notes
    clashing = set()
    for a, b in CLASHES:
        if a in cnotes and b in context:
            clashing.add(b)
        elif b in cnotes and a in context:
            clashing.add(a)
    penalty = min(0.5, _CLASH_PENALTY * len(clashing))
    if clashing:
        reasons.append("beißt sich mit " + ", ".join(sorted(_label(c) for c in clashing)))

    # Without a wish list there is nothing to hit, so cohesion carries the score.
    raw = (0.65 * want + 0.35 * bridge) if desired else bridge
    return max(0.0, min(1.0, raw - penalty)), reasons


def recommend(category, ingredients, ratios, context, desired, alcohol_free=False):
    """Rank the available ingredients for one role, best first.

    `context` are the ingredients already in the glass (the other roles' picks
    and any extras). Every candidate is scored against the same context, so the
    fit numbers are comparable within the role even while the drink is still
    half-empty.
    """
    context_notes = set()
    for i in context:
        context_notes |= notes_of(i)

    out = []
    for cand in candidates_for(category, ingredients, alcohol_free):
        fit = _fit(_dedupe(context + [cand]), ratios)
        flavor, reasons = flavor_score(cand, desired, context_notes, context)
        out.append({
            **_card(cand),
            "fit": round(fit, 3),
            "flavor": round(flavor, 3),
            "score": round(_FLAVOR_WEIGHT * flavor + (1 - _FLAVOR_WEIGHT) * fit, 3),
            "reasons": reasons,
        })
    out.sort(key=lambda r: -r["score"])
    return out


def balance(columns, ratios, axes=None):
    """Target vs. actually achieved amount per axis for the current selection.

    Axes the class never asked for are included when an ingredient pushes them
    above zero -- that unrequested bitter is the whole point of looking.
    """
    axes = axes or CATEGORIES
    if not columns:
        return {"axes": [], "fit": 0.0, "rel_residual": 0.0}

    x, rel = solve_amounts(columns, ratios)
    A = np.array([comp_vector(c, axes) for c in columns], float).T
    achieved = A @ x

    rows = []
    for i, axis in enumerate(axes):
        target = float(ratios.get(axis, 0.0))
        got = float(achieved[i])
        if target == 0.0 and got < 1e-6:
            continue
        rows.append({
            "axis": axis,
            "axis_label": ROLE_LABELS.get(axis, axis),
            "target": round(target, 2),
            "achieved": round(got, 2),
            "delta": round(got - target, 2),
        })
    return {"axes": rows, "fit": round(max(0.0, 1.0 - rel), 3), "rel_residual": round(rel, 3)}


def suggest_fixes(columns, ratios, ingredients, alcohol_free=False, limit=3):
    """Pure ingredients that would measurably improve an unreachable balance.

    When the picked ingredients can't hit the class ratios (too much rides along
    on a single ingredient), a pure one for the starved axis gives the solver the
    free axis it needs. Sloptails adds these behind the model's back; here we
    offer them and let the user decide.
    """
    base_fit = _fit(columns, ratios)
    if base_fit >= 1.0 - OVERCONSTRAINED_REL:
        return []

    have = {c["id"] for c in columns}
    out = []
    for axis in CATEGORIES:
        if not ratios.get(axis):
            continue
        for cand in candidates_for(axis, ingredients, alcohol_free):
            if cand["id"] in have or not is_pure(cand, axis):
                continue
            gain = _fit(columns + [cand], ratios) - base_fit
            if gain > 0.01:
                out.append({
                    **_card(cand),
                    "axis": axis,
                    "axis_label": ROLE_LABELS.get(axis, axis),
                    "gain": round(gain, 3),
                })
    out.sort(key=lambda r: -r["gain"])
    return out[:limit]


def build_state(class_def, ingredients, glasses, chosen_ids, extra_ids, notes,
                alcohol_free=False, name=""):
    """Everything the guided page needs for one render, derived from the picks.

    Stateless by design: the client owns the selection and posts it back, we
    re-solve and hand back roles, balance and the resulting drink.
    """
    ratios = class_def.get("ratios", {}) or {}
    ing_by_id = {i["id"]: i for i in ingredients}
    desired = {norm_note(n) for n in notes if str(n).strip()}
    roles = [c for c in CATEGORIES if c in ratios]

    # Only honour a pick that really can play the role it was picked for.
    picked = {}
    for role in roles:
        ing = ing_by_id.get(str(chosen_ids.get(role) or ""))
        if ing and ing.get("cat") == role:
            picked[role] = ing

    seen = {i["id"] for i in picked.values()}
    extras = []
    for eid in extra_ids or []:
        ing = ing_by_id.get(str(eid))
        if ing and ing["id"] not in seen:
            extras.append(ing)
            seen.add(ing["id"])

    columns = _dedupe(list(picked.values()) + extras)

    role_views = []
    for role in roles:
        context = _dedupe([i for r, i in picked.items() if r != role] + extras)
        role_views.append({
            "category": role,
            "label": ROLE_LABELS.get(role, role),
            "target": float(ratios.get(role, 0.0)),
            "chosen": picked[role]["id"] if role in picked else None,
            "candidates": recommend(role, ingredients, ratios, context, desired, alcohol_free),
        })

    state = {
        "roles": role_views,
        "balance": balance(columns, ratios),
        "extras": [_card(i) for i in extras],
        "preview": None,
        "fixes": [],
        "redundant": [],
    }

    if columns:
        amounts, _ = solve_amounts(columns, ratios)
        cocktail = build_recipe(columns, amounts, class_def, glasses)
        cocktail["name"] = (str(name).strip() or "Eigenkreation")[:80]
        cocktail["class_name"] = class_def.get("name", "")
        cocktail["description"] = class_def.get("description", "")
        state["preview"] = enrich(cocktail, ing_by_id, glasses)
        state["fixes"] = suggest_fixes(columns, ratios, ingredients, alcohol_free)
        # A pick the solver gave no room to (another ingredient already covers
        # its axes) silently drops out of the recipe -- say so instead. Read the
        # survivors back off the recipe so this can never contradict it.
        poured = {i["id"] for i in cocktail["ingredients"]}
        state["redundant"] = [_card(c) for c in columns if c["id"] not in poured]

    return state
