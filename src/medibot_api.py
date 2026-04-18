#!/usr/bin/env python3
import uuid, time, random, sys, os
from datetime import datetime
from typing import Dict, List
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

try:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    BASE_DIR = os.getcwd()

MODEL_DIR = os.path.join(BASE_DIR, "models")

# Ensure folder exists
os.makedirs(MODEL_DIR, exist_ok=True)

sk_path = os.path.join(MODEL_DIR, "sk_model.joblib")
tf_path = os.path.join(MODEL_DIR, "tf_model.keras")

# ── Load MediBot engine ────────────────────────────────────────────────────
print("\n  [MediBot API] Initialising NLP + ML...")

from .medibot import (
    NLPEngine,
    MLEngine,
    CONDITIONS,
    FOLLOWUP_BANK,
    Condition  # Class must be imported for joblib/pickle
)

# CRITICAL: Prevent AttributeError by linking the class to the current module
import __main__
__main__.Condition = Condition

NLP = NLPEngine()
ML = MLEngine(CONDITIONS)

def _init_engines():
    """
    Loads models and manually overrides the 'trained' state 
    to prevent the RuntimeError: 'Model not trained yet'.
    """
    from tensorflow import keras # type: ignore

    tokenizer_path = os.path.join(MODEL_DIR, "keras_tokenizer.joblib")
    maxlen_path = os.path.join(MODEL_DIR, "max_len.joblib")
    label_path = os.path.join(MODEL_DIR, "label_encoder.joblib")

    # Check ALL required files
    if all([
        os.path.exists(sk_path),
        os.path.exists(tf_path),
        os.path.exists(tokenizer_path),
        os.path.exists(maxlen_path),
        os.path.exists(label_path),
    ]):
        print("  [MediBot API] Loading models from disk...", end=" ", flush=True)

        ML.sk_clf = joblib.load(sk_path)
        ML.tf_clf.model = keras.models.load_model(tf_path)

        ML.tf_clf.keras_tokenizer = joblib.load(tokenizer_path)
        ML.tf_clf.MAX_LEN = joblib.load(maxlen_path)
        ML.tf_clf.label_encoder = joblib.load(label_path)

        ML.tf_clf._fitted = True

        print("Ready")

    else:
        print("  [MediBot API] Models not found. Training once...", end=" ", flush=True)

        # Train both models
        ML.train()

        # Save sklearn
        joblib.dump(ML.sk_clf, sk_path)

        # Save TF model
        ML.tf_clf.model.save(tf_path)

        # Save ALL required components
        joblib.dump(ML.tf_clf.keras_tokenizer, tokenizer_path)
        joblib.dump(ML.tf_clf.MAX_LEN, maxlen_path)
        joblib.dump(ML.tf_clf.label_encoder, label_path)

        print("Trained & Saved")

# Initialize models before starting the server
_init_engines()

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

MAX_SESSIONS = 500
SESSION_TTL  = 60 * 30 
SESSIONS: Dict[str, dict] = {}
MAX_FOLLOWUPS = 3

def _new_session():
    return {
        "phase":            "initial",
        "symptom_text":     [],
        "follow_up_count":  0,
        "asked_categories": [],
        "created_at":       time.time(),
        "last_active":      time.time(),
    }

def _get(sid):
    s = SESSIONS.get(sid)
    if s: s["last_active"] = time.time()
    return s

def _evict():
    now = time.time()
    for k in [k for k, v in SESSIONS.items() if now - v["last_active"] > SESSION_TTL]:
        del SESSIONS[k]

def _next_question(asked: List[str]) -> str:
    order = ["duration","severity","context","medications","associated"]
    for cat in order:
        if cat not in asked:
            asked.append(cat)
            return random.choice(FOLLOWUP_BANK[cat])
    return random.choice(FOLLOWUP_BANK[random.choice(order)])

def _build_results(ml_results, urgency, nlp_r):
    top = ml_results[0]
    return {
        "primaryCondition": top["name"],
        "risk":             top["probability_pct"],
        "urgency":          urgency,
        "duration":         nlp_r.get("duration") or "Not specified",
        "severitySignal":   "Severe" if nlp_r.get("severity",1.5) >= 2.5 else
                            "Moderate" if nlp_r.get("severity",1.5) >= 1.5 else "Mild",
        "conditions": [
            {
                "name": r["name"],
                "probability": r["probability_pct"],
                "severity": r["severity"],
                "description": r["description"]
            } for r in ml_results
        ],
        "recommendations": top["recommendations"],
        "disclaimer": "This analysis is not a substitute for a medical professional's diagnosys. If symptoms worsen, seek immediate medical attention. Always consult a qualified healthcare professional.",
    }

def _process(sess, message):
    phase = sess["phase"]

    if phase == "initial":
        nlp_r = NLP.preprocess(message)
        sess["symptom_text"].append(message)
        if not nlp_r["nltk_tokens"]:
            return {"reply": "I didn't catch specific symptoms. How are you feeling?", "phase": "initial", "results": None}
        
        q = _next_question(sess["asked_categories"])
        sess["follow_up_count"] = 1
        sess["phase"] = "gathering"
        return {"reply": "Let me ask a few follow-up questions.\n\n" + q, "phase": "gathering", "results": None}

    if phase == "gathering":
        sess["symptom_text"].append(message)
        if sess["follow_up_count"] < MAX_FOLLOWUPS:
            q = _next_question(sess["asked_categories"])
            sess["follow_up_count"] += 1
            return {"reply": q, "phase": "gathering", "results": None}

        # Analyze
        sess["phase"] = "analyzing"
        full = " ".join(sess["symptom_text"])
        nlp_r = NLP.preprocess(full)
        ml_results = ML.predict(full, nlp_r, top_n=5)
        
        u_score = {"Low":1, "Medium":2, "High":3}
        urgency = {1:"Low", 2:"Medium", 3:"High"}[max(u_score[r["urgency"]] for r in ml_results[:3])]
        payload = _build_results(ml_results, urgency, nlp_r)
        
        sess["phase"] = "results"
        return {"reply": "Analysis complete.", "phase": "results", "results": payload}

    return {"reply": "Analysis complete.", "phase":"results","results":None}

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.route("/api/session", methods=["POST"])
def create_session():
    _evict()
    if len(SESSIONS) >= MAX_SESSIONS:
        return jsonify({"error": "Server at capacity."}), 503
    sid = str(uuid.uuid4())
    SESSIONS[sid] = _new_session()
    return jsonify({"sessionId": sid, "greeting": "Hello! I'm your AI health assistant. What symptoms are you experiencing today? List symptoms separated by commas - e.g. fever, headache, stiff neck"}), 201

@app.route("/api/chat", methods=["POST"])
def chat():
    body = request.get_json(silent=True) or {}
    sid = body.get("sessionId","").strip()
    message = body.get("message","").strip()
    
    if not sid or not message:
        return jsonify({"error":"sessionId and message required"}), 400
        
    sess = _get(sid)
    if sess is None:
        return jsonify({"error":"Session not found or expired."}), 404
        
    result = _process(sess, message)
    result["sessionId"] = sid
    return jsonify(result), 200

@app.route("/api/reset/<sid>", methods=["POST"])
def reset_session(sid):
    if sid not in SESSIONS: return jsonify({"error":"Session not found"}), 404
    SESSIONS[sid] = _new_session()
    return jsonify({"sessionId":sid,"phase":"initial",
                    "greeting":"Hello again! Describe your symptoms for a fresh analysis."}), 200

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "activeSessions": len(SESSIONS)}), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)