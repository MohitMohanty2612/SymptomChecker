#!/usr/bin/env python3
import uuid, time, random, sys, os
from datetime import datetime
from typing import Dict, List
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import src.medibot as medibot

sys.modules['medibot'] = medibot


try:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    BASE_DIR = os.getcwd()

MODEL_DIR = os.path.join(BASE_DIR, "models")

# Ensure folder exists
os.makedirs(MODEL_DIR, exist_ok=True)

sk_path = os.path.join(MODEL_DIR, "ml_model.joblib")

# ── Load MediBot engine ────────────────────────────────────────────────────
print("\n  [MediBot API] Initialising NLP + ML...")

from medibot import (
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

MODELS_LOADED = False

def load_models_once():
    global MODELS_LOADED
    if not MODELS_LOADED:
        print("Loading models (first time only)...")
        _init_engines()
        MODELS_LOADED = True

def _init_engines():
    global ML
    print("🔥 INIT ENGINES CALLED")
    if os.path.exists(sk_path):
        print("  [MediBot API] Loading sklearn model...", end=" ", flush=True)
        ML = joblib.load(sk_path)
        print("Ready")
    else:
        print("  [MediBot API] Training sklearn model...", end=" ", flush=True)
        ML.train()   # Make sure this trains ONLY sklearn now
        joblib.dump(ML, sk_path)
        print("Trained & Saved")

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
        "symptoms":         [],
        "meta":             [],
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
    all_categories = ["duration","severity","context","medications","associated"]
    remaining = [c for c in all_categories if c not in asked]
    if not remaining:
        remaining = all_categories
    cat = random.choice(remaining)
    asked.append(cat)
    return random.choice(FOLLOWUP_BANK[cat])

def _build_results(ml_results, urgency, nlp_r):
    if not ml_results:
        return {
            "primaryCondition": "Unknown",
            "risk": "0%",
            "urgency": "Low",
            "conditions": [],
            "recommendations": ["Consult a doctor"],
            "disclaimer": "No prediction available"
        }
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
        "disclaimer": "This analysis is not a substitute for a medical professional's diagnosys.\nIf symptoms worsen, seek immediate medical attention.\nAlways consult a qualified healthcare professional.",
    }

def _process(sess, message):
    phase = sess["phase"]

    if phase == "initial":
        nlp_r = NLP.preprocess(message)
        sess["symptoms"].append(message)
        if not nlp_r["nltk_tokens"]:
            return {"reply": "I didn't catch specific symptoms. How are you feeling?", "phase": "initial", "results": None}
        
        q = _next_question(sess["asked_categories"])
        sess["follow_up_count"] = 1
        sess["phase"] = "gathering"
        return {"reply": "Let me ask a few follow-up questions.\n\n" + q, "phase": "gathering", "results": None}

    if phase == "gathering":
        sess["meta"].append(message)
        if sess["follow_up_count"] < MAX_FOLLOWUPS:
            q = _next_question(sess["asked_categories"])
            sess["follow_up_count"] += 1
            return {"reply": q, "phase": "gathering", "results": None}

        # Analyze
        sess["phase"] = "analyzing"
        full = " ".join(sess["symptoms"] + sess["meta"])
        print("SYMPTOMS:", sess["symptoms"])
        print("META:", sess["meta"])
        print("MODEL INPUT:", full)
        nlp_r = NLP.preprocess(full)
        user_symptoms = set(nlp_r.get("affirmed", []))
        # 🔥 Get ALL ML predictions (not just top 3)
        # 1. Get ALL predictions
        ml_results = ML.predict(full, nlp_r, top_n=len(CONDITIONS))

        user_symptoms = set([s.lower() for s in nlp_r.get("affirmed", [])] + [full.lower()])

        matched_results = []

        # 2. Strict symptom matching
        for r in ml_results:
            cond_obj = next((c for c in CONDITIONS if c.name == r["name"]), None)
            if not cond_obj:
                continue

            def normalize(text):
                return text.lower().replace("_", " ").strip()

            user_symptoms_norm = set([normalize(s) for s in user_symptoms])
            cond_symptoms_norm = set([normalize(s) for s in cond_obj.symptoms])

            # ✅ smarter matching
            overlap = 0
            for us in user_symptoms_norm:
                for cs in cond_symptoms_norm:
                    # exact phrase match (VERY IMPORTANT)
                    if us == cs:
                        overlap += 3

                    # partial phrase match
                    elif us in cs or cs in us:
                        overlap += 1

            if overlap > 0:
                r["match_score"] = overlap
                matched_results.append(r)

        # 3. Sort properly
        matched_results = sorted(
            matched_results,
            key=lambda x: float(x["probability_pct"].replace('%','')),
            reverse=True
        )

        # 4. ✅ FINAL: take TOP 5 ONLY
        matched_results = matched_results[:5]
        
        u_score = {"Low":1, "Medium":2, "High":3}
        urgency = {1:"Low", 2:"Medium", 3:"High"}[max(u_score[r["urgency"]] for r in matched_results)]
        payload = _build_results(matched_results, urgency, nlp_r)
        
        sess["phase"] = "results"
        return {"reply": "Analysis complete.", "phase": "results", "results": payload}

    return {"reply": "Analysis complete.", "phase":"results","results":None}

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "MediBot API is running.",
        "endpoints": [
            "/api/session",
            "/api/chat",
            "/api/conditions",
            "/api/health"
        ]
    }), 200
    
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
    try:
        load_models_once()
        body = request.get_json(silent=True) or {}
        sid = body.get("sessionId","").strip()
        message = body.get("message","").strip()
        
        if not sid or not message:
            return jsonify({"error":"sessionId and message required"}), 400
            
        sess = _get(sid)
        if sess is None:
            return jsonify({"error":"Session not found or expired."}), 404
            
        try:
            result = _process(sess, message)
            result["sessionId"] = sid
            return jsonify(result), 200

        except Exception as e:
            import traceback
            print("🔥 ERROR IN /api/chat:")
            traceback.print_exc()

            return jsonify({
                "error": str(e)
            }), 500
    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({
            "error": "Internal server error",
            "details": str(e)
        }), 500

@app.route("/api/reset/<sid>", methods=["POST"])
def reset_session(sid):
    if sid not in SESSIONS: return jsonify({"error":"Session not found"}), 404
    SESSIONS[sid] = _new_session()
    return jsonify({"sessionId":sid,"phase":"initial",
                    "greeting":"Hello again! Describe your symptoms for a fresh analysis."}), 200

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "activeSessions": len(SESSIONS), "modelsLoaded": MODELS_LOADED}), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
