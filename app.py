# -*- coding: utf-8 -*-
"""
AgriHealth AI – Backend (FastAPI)
---------------------------------------------------------------------
Website (js/detect.js) yahan 2 cheezein bhejti hai:
  POST /predict/photo   photo + fasal  -> bimari, confidence, heatmap
  POST /predict/text    alamaat/jumla  -> mumkina bimariyan
Aur:
  GET  /                test page (browser mein photo daal kar check karein)
  GET  /health          server theek chal raha hai ya nahi + model ke results
  GET  /docs            FastAPI ka apna test page

Privacy: tasveerein aur jumle sirf jawab banane ke liye memory mein rehte hain.
Kuch bhi save ya log nahi hota (website ki Privacy Policy ke mutabiq).
"""
import io
import json
import os
import threading
import time
from collections import defaultdict, deque
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field

import engine
import model_runtime

HERE = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------------ settings
MAX_UPLOAD_MB = 10
MIN_SIDE_PX = 64                  # website 200px se chhoti photo pehle hi rok deti hai
RATE_LIMIT = int(os.environ.get("RATE_LIMIT_PER_MINUTE", "40"))
# Domain lene ke baad Space ki Settings mein ALLOWED_ORIGINS = https://www.aapkadomain.com rakh dein
ALLOWED_ORIGINS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",") if o.strip()]

Image.MAX_IMAGE_PIXELS = 40_000_000           # bohat bari (khatarnak) tasveerein band
THREADS = int(os.environ.get("TORCH_THREADS", "2"))

# ------------------------------------------------------------------ model
with open(os.path.join(HERE, "config.json"), encoding="utf-8") as f:
    CONFIG = json.load(f)
CLASSES = CONFIG["classes"]
CROPS = CONFIG["crops"]
SLUGS = dict(CONFIG["slugs"])
# Model ki kuch classes ke liye ab website par apne guide hain (Phase 1). Pehle yeh "other" thin.
NEW_GUIDE_SLUGS = {
    "rice__brown_spot": "rice-brown-spot",
    "tomato__yellow_leaf_curl": "tomato-leaf-curl-virus",
}
for _cls, _slug in NEW_GUIDE_SLUGS.items():
    if _cls in SLUGS:
        SLUGS[_cls] = _slug
SUPPORTED_CROPS = sorted({c for c in CROPS.values() if c})

# model.onnx ho to halka ONNX model (onnxruntime), warna model.pt (PyTorch). Jawab dono ka ek jaisa.
MODEL = model_runtime.load_model(HERE, CONFIG, THREADS)
MODEL_LOCK = threading.Lock()      # free server par ek waqt mein ek tasveer (memory bachane ke liye)

with open(os.path.join(HERE, "diseases.json"), encoding="utf-8") as f:
    MATCHER = engine.TextMatcher(json.load(f))

print(f"Model ready ({MODEL.kind}): {CONFIG['model_name']} | {len(CLASSES)} classes | crops: {', '.join(SUPPORTED_CROPS)}", flush=True)

# ------------------------------------------------------------------ app
app = FastAPI(title="AgriHealth AI API", version="1.0",
              description="Plant disease photo check and symptom matching for the AgriHealth AI website.")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------ simple rate limit (per IP)
_hits = defaultdict(deque)
_hits_lock = threading.Lock()


def check_rate(request: Request):
    forwarded = request.headers.get("x-forwarded-for", "")
    ip = forwarded.split(",")[0].strip() or (request.client.host if request.client else "unknown")
    now = time.time()
    with _hits_lock:
        queue = _hits[ip]
        while queue and now - queue[0] > 60:
            queue.popleft()
        if len(queue) >= RATE_LIMIT:
            raise HTTPException(429, "Too many requests. Please wait a minute and try again.")
        queue.append(now)
        if len(_hits) > 5000:          # purane IPs saaf
            for key in [k for k, q in _hits.items() if not q or now - q[-1] > 60]:
                del _hits[key]


def check_type_host(kind, host):
    if kind not in ("plant", "animal"):
        raise HTTPException(422, "type must be 'plant' or 'animal'")
    if not MATCHER.valid_host(kind, host):
        raise HTTPException(422, f"Unknown {kind}: {host}")


# ------------------------------------------------------------------ photo
@app.post("/predict/photo")
def predict_photo(request: Request,
                  image: UploadFile = File(...),
                  type: str = Form("plant"),
                  host: str = Form(...)):
    check_rate(request)
    check_type_host(type, host)

    if type == "animal" or host not in SUPPORTED_CROPS:
        # Is fasal/janwar ki photo ka model abhi tayyar nahi: website "likh kar batayein" ka mashwara deti hai
        return {"supported": False, "valid_image": True, "predictions": []}

    raw = image.file.read(MAX_UPLOAD_MB * 1024 * 1024 + 1)
    if len(raw) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"Photo is larger than {MAX_UPLOAD_MB} MB.")
    try:
        img = Image.open(io.BytesIO(raw))
        img = ImageOps.exif_transpose(img).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        raise HTTPException(400, "This file is not a readable photo.")
    if min(img.size) < MIN_SIDE_PX:
        raise HTTPException(400, "This photo is too small.")

    x = engine.preprocess(img, CONFIG["img_size"], CONFIG["mean"], CONFIG["std"])
    with MODEL_LOCK:
        features, logits = MODEL.run(x)
    answer = engine.photo_answer(logits, CLASSES, CROPS, SLUGS,
                                 CONFIG["temperature"], CONFIG["not_leaf_limit"], host)
    heatmap = None
    if answer["valid_image"]:
        heatmap = engine.heatmap_data_url(MODEL.cam(features, answer["top_index"]), img.size)

    del answer["top_index"]
    answer["supported"] = True
    if heatmap:
        answer["heatmap"] = heatmap
    answer["model"] = CONFIG.get("trained_on", "")
    return answer


# ------------------------------------------------------------------ text (symptoms)
class TextRequest(BaseModel):
    type: str = "plant"
    host: str
    symptoms: List[str] = Field(default_factory=list, max_length=40)
    text: str = Field("", max_length=1000)
    duration: Optional[str] = None
    count: Optional[str] = None


@app.post("/predict/text")
def predict_text(body: TextRequest, request: Request):
    check_rate(request)
    check_type_host(body.type, body.host)
    predictions = MATCHER.match(body.type, body.host, body.symptoms, body.text)
    return {"predictions": predictions, "engine": "symptom-similarity"}


# ------------------------------------------------------------------ status + test page
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": CONFIG["model_name"],
        "runtime": MODEL.kind,
        "trained_on": CONFIG.get("trained_on", ""),
        "photo_crops": SUPPORTED_CROPS,
        "classes": len(CLASSES),
        "test_results": CONFIG.get("results", {}),
    }


TEST_PAGE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>AgriHealth AI API – test</title>
<style>
 body{font-family:system-ui,sans-serif;max-width:720px;margin:32px auto;padding:0 16px;color:#13241c;background:#f6faf7}
 h1{color:#0b3b2c;font-size:24px} .box{background:#fff;border:1px solid #dbe7df;border-radius:14px;padding:20px;margin-top:16px}
 button{background:#1a8a4a;color:#fff;border:0;border-radius:10px;padding:10px 18px;font-weight:700;cursor:pointer}
 select,input{margin:6px 0 12px;font-size:15px} .wrap{position:relative;display:inline-block;max-width:100%}
 .wrap img{max-width:100%;border-radius:10px;display:block} #heat{position:absolute;inset:0;width:100%;height:100%;opacity:.6}
 pre{white-space:pre-wrap;background:#0b3b2c;color:#d8f5e3;padding:14px;border-radius:10px;font-size:13px}
</style></head><body>
<h1>AgriHealth AI API is running ✅</h1>
<p>Test the photo model here. The website uses the same address. Photos are not stored.</p>
<div class="box">
 <label>Crop</label><br><select id="host">__OPTIONS__</select><br>
 <label>Leaf photo</label><br><input type="file" id="file" accept="image/*"><br>
 <button id="go">Check photo</button>
</div>
<div class="box" id="out" hidden>
 <div class="wrap"><img id="photo" alt="Your photo"><img id="heat" alt="" hidden></div>
 <pre id="json"></pre>
</div>
<p>More: <a href="/health">/health</a> · <a href="/docs">/docs</a></p>
<script>
document.getElementById("go").onclick = async function () {
  const file = document.getElementById("file").files[0];
  if (!file) { alert("Choose a photo first"); return; }
  const form = new FormData();
  form.append("image", file); form.append("type", "plant"); form.append("host", document.getElementById("host").value);
  document.getElementById("json").textContent = "Checking...";
  document.getElementById("out").hidden = false;
  document.getElementById("photo").src = URL.createObjectURL(file);
  const heat = document.getElementById("heat"); heat.hidden = true;
  const res = await fetch("/predict/photo", { method: "POST", body: form });
  const data = await res.json();
  if (data.heatmap) { heat.src = data.heatmap; heat.hidden = false; data.heatmap = data.heatmap.slice(0, 40) + "..."; }
  document.getElementById("json").textContent = JSON.stringify(data, null, 2);
};
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def home():
    options = "".join(f'<option value="{c}">{c.title()}</option>' for c in SUPPORTED_CROPS)
    return TEST_PAGE.replace("__OPTIONS__", options)
