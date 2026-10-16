# -*- coding: utf-8 -*-
"""
AgriHealth AI – engine.py
Model ke bahar ka saara hisaab: tasveer tayyar karna, model ke numbers se jawab banana,
Grad-CAM heatmap ki tasveer, aur symptoms (likh kar batana) wala matcher.
Is file ko torch ki zaroorat nahi, is liye ise alag se test kiya ja sakta hai.
"""
import base64
import io
import re

import numpy as np
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer

UNCERTAIN_BELOW = 0.5     # is se kam confidence ho to website "pakka nahi" ka note dikhati hai
CLEAR_CLASS_BELOW = 0.35  # "other" jaise milay jule jawab mein bhi koi ek bimari saaf honi chahiye
MAX_PREDICTIONS = 3


# ------------------------------------------------------------------ photo: input
def preprocess(img, size, mean, std):
    """PIL image -> (1, 3, size, size) float32, bilkul training (eval_tf) jaisa."""
    img = img.convert("RGB").resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    return np.ascontiguousarray(arr.transpose(2, 0, 1)[None])


def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()


# ------------------------------------------------------------------ photo: answer
def photo_answer(logits, classes, crops, slugs, temperature, not_leaf_limit, crop):
    """
    logits: model ka output (num_classes,)
    Jawab: {"valid_image", "predictions": [{"slug", "confidence"}], "uncertain", "top_index"}
    """
    probs = softmax(np.asarray(logits, dtype=np.float64) / float(temperature))
    if "not_leaf" in classes and probs[classes.index("not_leaf")] > not_leaf_limit:
        return {"valid_image": False, "predictions": [], "uncertain": False, "top_index": None}

    crop_idx = [i for i, c in enumerate(classes) if crops.get(c) == crop]
    p = probs[crop_idx]
    p = p / p.sum()

    by_slug = {}
    for j, i in enumerate(crop_idx):
        slug = slugs[classes[i]]
        by_slug[slug] = by_slug.get(slug, 0.0) + float(p[j])
    preds = sorted(({"slug": s, "confidence": round(v, 4)} for s, v in by_slug.items()),
                   key=lambda d: -d["confidence"])[:MAX_PREDICTIONS]
    return {
        "valid_image": True,
        "predictions": preds,
        # "pakka nahi": sab se upar wala jawab kamzor ho, ya koi ek class bhi saaf aage na ho
        "uncertain": preds[0]["confidence"] < UNCERTAIN_BELOW or float(p.max()) < CLEAR_CLASS_BELOW,
        "top_index": crop_idx[int(np.argmax(p))],   # heatmap isi class ka banta hai
    }


# ------------------------------------------------------------------ photo: heatmap
def grad_cam(features, gradients):
    """features, gradients: (C, h, w) -> cam (h, w) 0..1"""
    weights = gradients.mean(axis=(1, 2))
    cam = np.maximum((weights[:, None, None] * features).sum(axis=0), 0)
    cam = cam - cam.min()
    peak = cam.max()
    return cam / peak if peak > 1e-8 else cam


def jet(values):
    """0..1 -> RGB (jet colormap), matplotlib ke baghair."""
    v = np.clip(values, 0, 1)
    r = np.clip(1.5 - np.abs(4 * v - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * v - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * v - 1), 0, 1)
    return np.stack([r, g, b], axis=-1)


def heatmap_data_url(cam, image_size, max_side=512):
    """cam (h, w) -> transparent PNG (data URL), photo ke barabar shakal mein."""
    w, h = image_size
    scale = min(1.0, max_side / max(w, h))
    out_w, out_h = max(1, int(w * scale)), max(1, int(h * scale))
    big = Image.fromarray(np.asarray(cam, dtype=np.float32), mode="F").resize((out_w, out_h), Image.BILINEAR)
    values = np.clip(np.asarray(big), 0, 1)
    rgba = np.zeros((out_h, out_w, 4), dtype=np.uint8)
    rgba[..., :3] = (jet(values) * 255).astype(np.uint8)
    rgba[..., 3] = (values * 200).astype(np.uint8)      # thanda hissa transparent
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


# ------------------------------------------------------------------ text: symptoms
# Roman Urdu mein ek hi lafz kai tarah likha jata hai (laal/lal, gaanthein/ganthein),
# is liye lambi awaazein chhoti kar ke milate hain. Yeh chhote aam alfaaz kisi bimari ki nishani nahi.
STOP_WORDS = {
    "ka", "ki", "ke", "ko", "se", "me", "mein", "main", "par", "pe", "aur", "or", "hai", "hain", "he", "ho", "hy",
    "nahi", "nhi", "nahin", "raha", "rahi", "rahe", "gaya", "gayi", "gai", "gaye", "wala", "wali", "wale", "bhi",
    "the", "and", "of", "on", "in", "is", "are", "a", "an", "to", "my", "has", "have", "with",
}


def normalize(text):
    text = (text or "").lower()
    for long, short in (("aa", "a"), ("ee", "i"), ("oo", "u"), ("ii", "i")):
        text = text.replace(long, short)
    return text


def words_of(text):
    return [w for w in re.findall(r"\w+", normalize(text)) if w not in STOP_WORDS and len(w) >= 2]


def keyword_score(keyword, text, text_words):
    """1 = poora keyword mila, 0.5 = lambe keyword ke aadhe (kam az kam 2) alfaaz mile, 0 = nahi mila."""
    if keyword.lower() in text:
        return 1.0
    parts = words_of(keyword)
    if not parts:
        return 0.0
    hits = sum(1 for p in parts if any(w == p or (len(p) >= 4 and w.startswith(p[:4])) for w in text_words))
    if hits == len(parts):
        return 1.0
    if hits >= 2 and hits * 2 >= len(parts):     # kam az kam 2 alfaaz mile hon (aik aam lafz kaafi nahi)
        return 0.5
    return 0.0


class TextMatcher:
    """
    Symptoms ka check:
      - chuni hui alamaat (chips) jo bimari se milti hain   -> 2 points har ek
      - keywords (English, Roman Urdu, Urdu) jo likhe gaye  -> 1 point har ek (zyada se zyada 4);
        Roman Urdu ki mukhtalif spelling (laal/lal) aur aadha mila keyword (0.5) bhi ginte hain
      - likhe hue jumle ki bimari ki tafseel se mushabahat  -> 0 se 4 points
        (TF-IDF character n-grams: English, Urdu aur Roman Urdu teeno par chalta hai,
         aur ghalat spelling par bhi kuch had tak kaam karta hai)
    """

    MIN_SIMILARITY = 0.08
    MIN_SCORE = 0.8
    MAX_CONFIDENCE = 0.9      # yeh basic matching hai, is liye 90% se zyada kabhi nahi

    def __init__(self, data):
        self.diseases = data["diseases"]
        self.hosts = data["hosts"]
        docs = [d["doc"] + " " + " ".join(d["keywords"]) for d in self.diseases]
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), lowercase=True, sublinear_tf=True)
        self.doc_vectors = self.vectorizer.fit_transform(docs)

    def valid_host(self, kind, host):
        return host in self.hosts.get(kind, [])

    def match(self, kind, host, symptoms, text):
        text = (text or "").strip().lower()
        text_words = words_of(text)
        chips = set(symptoms or [])
        index = [i for i, d in enumerate(self.diseases) if d["type"] == kind and host in d["hosts"]]
        if not index:
            return []

        sims = np.zeros(len(self.diseases))
        if len(text) >= 3:
            sims = (self.doc_vectors @ self.vectorizer.transform([text]).T).toarray().ravel()

        scored = []
        for i in index:
            d = self.diseases[i]
            score = 2.0 * len(chips & set(d["chips"]))
            if text:
                score += min(sum(keyword_score(word, text, text_words) for word in d["keywords"]), 4)
                if sims[i] >= self.MIN_SIMILARITY:
                    score += 4.0 * float(sims[i])
            if score >= self.MIN_SCORE:
                scored.append((d["slug"], score))

        if not scored:
            return []
        scored.sort(key=lambda item: -item[1])
        total = sum(s for _, s in scored)
        return [{"slug": slug, "confidence": round(min(self.MAX_CONFIDENCE, s / total), 4)}
                for slug, s in scored[:4]]
