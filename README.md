# AgriHealth AI – Backend (halka / ONNX version)

Yeh wahi backend hai jo Hugging Face wala hai, magar PyTorch ke baghair. Model ONNX format mein chalta hai,
is liye 512 MB RAM wale free server (jaise **Render**) par bhi chal jata hai. Jawab aur heatmap bilkul ek jaise hain.

| Address | Kaam |
|---|---|
| `GET /` | Test page: patte ki photo daal kar AI ka jawab + heatmap dekhein |
| `GET /health` | Server theek hai ya nahi, model ke test results (`"runtime": "onnx"`) |
| `POST /predict/photo` | `image`, `type` (`plant`), `host` (`tomato`, `potato`, `rice`) → bimari, confidence, heatmap |
| `POST /predict/text` | JSON `{type, host, symptoms[], text}` → mumkina bimariyan (English, Urdu, Roman Urdu) |

## Files (sab ek hi folder mein, kisi sub-folder mein nahi)

- Is zip se: `app.py`, `engine.py`, `model_runtime.py`, `diseases.json`, `Dockerfile`, `requirements.txt`, `README.md`
- `agrihealth_model_onnx.zip` (Kaggle export notebook) se: `model.onnx`, `head.npz`, `config.json`

## Settings (Environment variables, zaroori nahi)

- `ALLOWED_ORIGINS` – domain lene ke baad `https://www.aapkadomain.com`, taake sirf aap ki website API use kare.
- `RATE_LIMIT_PER_MINUTE` – har visitor ke liye ek minute mein kitne check (default 40).

**Privacy:** tasveerein aur jumle sirf jawab banane ke liye memory mein rehte hain. Kuch save ya log nahi hota.
