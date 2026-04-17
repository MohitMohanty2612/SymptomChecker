#!/usr/bin/env python3
"""
medibot_api.py — Flask REST API wrapping MediBot v2
Uses: NLTK + spaCy (NLP) · TensorFlow/Keras BiLSTM + sklearn (ML)

Run:
    python medibot_api.py

Endpoints :
    POST   /api/session
    POST   /api/chat
    POST   /api/reset/<sid>
    GET    /api/conditions
    GET    /api/health
    DELETE /api/session/<sid>
"""

import uuid, time, random, sys, os
from datetime import datetime
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify

# ── Load MediBot engine ────────────────────────────────────────────────────
print("\n  [MediBot API] Initialising NLP (NLTK + spaCy) + ML (TF + sklearn)...")
t0 = time.time()

from medibot import (
    NLPEngine,
    MLEngine,
    CONDITIONS,
    FOLLOWUP_BANK,
)

NLP = NLPEngine()

ML = MLEngine(CONDITIONS)

print("  [MediBot API] Training sklearn models...", end=" ", flush=True)
ML.sk_clf.fit(ML._train_docs, ML._train_labels)
print("✓")

print("  [MediBot API] Training TensorFlow BiLSTM...", end=" ", flush=True)
ML.tf_clf.fit(ML._train_docs, ML._train_labels, epochs=40, batch_size=16)
print(f"✓  ({time.time()-t0:.1f}s)")
print(f"  [MediBot API] {len(CONDITIONS)} conditions loaded. Server ready.\n")

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)

MAX_SESSIONS = 500
SESSION_TTL  = 60 * 30    # 30 min
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
        "affirmations":     list(nlp_r.get("affirmed", set()))[:8],
        "negations":        list(nlp_r.get("negated", set()))[:6],
        "spacyEntities":    [{"text":e["text"],"label":e["label"]}
                             for e in nlp_r.get("spacy_entities",[])[:5]],
        "conditions": [
            {
                "name":        r["name"],
                "icd":         r["icd"],
                "probability": r["probability_pct"],
                "severity":    r["severity"],
                "description": r["description"],
                "bodySystem":  r["body_system"],
            }
            for r in ml_results
        ],
        "recommendations": top["recommendations"],
        "disclaimer": (
            "This analysis is not a substitute for a medical professional's diagnosys. "
            "If symptoms worsen, seek immediate medical attention. "
            "Always consult a qualified healthcare professional."
        ),
        "nlpPipeline": "NLTK word_tokenize · pos_tag · WordNetLemmatizer · MWETokenizer · ne_chunk "
                       "+ spaCy en_core_web_sm NER · dep-parse negation · lemma_",
        "mlPipeline":  "TF Keras BiLSTM (40%) + sklearn TF-IDF CalibratedSVC + ComplementNB + Cosine (60%)",
    }

def _process(sess, message):
    phase = sess["phase"]

    if phase == "initial":
        nlp_r = NLP.preprocess(message)
        sess["symptom_text"].append(message)
        if not nlp_r["nltk_tokens"]:
            return {"reply": "I didn't catch specific symptoms. Could you describe how you're feeling?", "phase": "initial", "results": None}
        affirmed = [t for t in list(nlp_r["affirmed"])[:3] if len(t) > 3]
        ##ack = (f"I noticed you mentioned: {', '.join(affirmed[:2])}. " if affirmed else "Thank you. ")
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

        # Enough info — analyse
        sess["phase"] = "analyzing"
        full = " ".join(sess["symptom_text"])
        nlp_r = NLP.preprocess(full)
        ml_results = ML.predict(full, nlp_r, top_n=5)
        u_score = {"Low":1,"Medium":2,"High":3}
        urgency = {1:"Low",2:"Medium",3:"High"}[max(u_score[r["urgency"]] for r in ml_results[:3])]
        payload = _build_results(ml_results, urgency, nlp_r)
        sess["phase"] = "results"
        msg = {"High": "⚠️ HIGH URGENCY — please seek immediate medical attention.",
               "Medium": "Your symptoms suggest consulting a doctor soon.",
               "Low": "Symptoms appear manageable — see a doctor if they worsen."}[urgency]
        return {"reply": f"Analysing your symptoms... ",
                "phase": "results", "results": payload}

    return {"reply": "Analysis complete. Click 'Start Over' to check new symptoms.", "phase":"results","results":None}


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.route("/api/session", methods=["POST"])
def create_session():
    _evict()
    if len(SESSIONS) >= MAX_SESSIONS:
        return jsonify({"error": "Server at capacity."}), 503
    sid = str(uuid.uuid4())
    SESSIONS[sid] = _new_session()
    return jsonify({
        "sessionId": sid,
        "greeting":  ("Hello! I'm your AI health assistant. What symptoms are you experiencing today? "
                      "List symptoms separated by commas - e.g. fever, headache, stiff neck"),
        "phase":     "initial",
        "conditionCount": len(CONDITIONS),
        "nlpEngines": ["NLTK", "spaCy en_core_web_sm"],
        "mlEngines":  ["TensorFlow/Keras BiLSTM", "sklearn TF-IDF+SVC", "sklearn ComplementNB", "Cosine Similarity"],
    }), 201

@app.route("/api/chat", methods=["POST"])
def chat():
    body    = request.get_json(silent=True) or {}
    sid     = body.get("sessionId","").strip()
    message = body.get("message","").strip()
    if not sid:     return jsonify({"error":"sessionId required"}), 400
    if not message: return jsonify({"error":"message required"}), 400
    sess = _get(sid)
    if sess is None: return jsonify({"error":"Session not found or expired."}), 404
    result = _process(sess, message)
    result["sessionId"] = sid
    return jsonify(result), 200

@app.route("/api/reset/<sid>", methods=["POST"])
def reset_session(sid):
    if sid not in SESSIONS: return jsonify({"error":"Session not found"}), 404
    SESSIONS[sid] = _new_session()
    return jsonify({"sessionId":sid,"phase":"initial",
                    "greeting":"Hello again! Describe your symptoms for a fresh analysis."}), 200

@app.route("/api/session/<sid>", methods=["DELETE"])
def delete_session(sid):
    SESSIONS.pop(sid, None)
    return jsonify({"message":"Deleted."}), 200

@app.route("/api/conditions", methods=["GET"])
def list_conditions():
    return jsonify({
        "total": len(CONDITIONS),
        "conditions": [
            {"name":c.name,"icd":c.icd,"severity":c.severity,
             "urgency":c.urgency,"bodySystem":c.body_system,"description":c.description}
            for c in CONDITIONS
        ],
    }), 200

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "activeSessions": len(SESSIONS),
        "conditionsLoaded": len(CONDITIONS),
        "nlpEngines": ["NLTK","spaCy"],
        "mlEngines":  ["TensorFlow/Keras","sklearn"],
        "modelsReady": True,
        "timestamp": datetime.utcnow().isoformat()+"Z",
    }), 200

@app.after_request
def cors(r):
    r.headers["Access-Control-Allow-Origin"]  = "*"
    r.headers["Access-Control-Allow-Headers"] = "Content-Type"
    r.headers["Access-Control-Allow-Methods"] = "GET,POST,DELETE,OPTIONS"
    return r

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"  [MediBot API] http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False)