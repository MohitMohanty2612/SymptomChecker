# Standard Imports
import json
import os
import re, random
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Set
import numpy as np
from scipy.special import softmax

# NLP Imports

import spacy
from spacy.tokens import Doc
from spacy.lang.en.stop_words import STOP_WORDS

# ML Imports

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline

# Medical Knowledge Base

@dataclass
class Condition:
    name: str
    icd: str
    symptoms: List[str]
    keywords: List[str]
    severity: str
    urgency: str
    description: str
    recommendations: List[str]
    body_system: str

def load_conditions():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    json_path = os.path.join(BASE_DIR, "data", "conditions.json")

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"conditions.json not found at {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return [
        Condition(
            name=item["name"],
            icd=item["icd"],
            symptoms=item["symptoms"],
            keywords=item["keywords"],
            severity=item["severity"],
            urgency=item["urgency"],
            description=item["description"],
            recommendations=item["recommendations"],
            body_system=item["body_system"],
        )
        for item in data
    ]

CONDITIONS: List[Condition] = load_conditions()

# NLP Engine

# Medical synonym normalisation map
SYNONYMS: Dict[str, str] = {
    "tummy":"stomach","belly":"stomach","gut":"stomach",
    "throat":"throat","pharynx":"throat",
    "chest":"chest","thorax":"chest","breast":"chest","boobs":"breast",
    "cardiac":"heart","cardio":"heart",
    "gastric":"stomach","abdominal":"stomach abdomen",
    "lumbar":"back","spine":"back","spinal":"back","ass":"lower back",
    "articular":"joint",
    "myalgia":"muscle pain",
    "breath":"breathing","respiratory":"breathing",
    "urinate":"urination","pee":"urination","urine":"urination",
    "puke":"vomiting","nausea":"nausea",
    "dizzy":"dizziness","lightheaded":"dizziness","vertigo":"dizziness",
    "tired":"fatigue","exhausted":"fatigue","lethargic":"fatigue",
    "temp":"fever","feverish":"fever","pyrexia":"fever",
    "itching": "itchy skin","itchy": "itchy skin",
    "skin itching": "itchy skin","itching skin": "itchy skin","skin rash": "rash",
    "swollen":"swelling","puffiness":"swelling","edema":"swelling",
    "phlegm":"mucus","sputum":"mucus",
    "heartburn":"acid reflux","indigestion":"dyspepsia","bloated":"bloating",
    "palpitation":"heart palpitations","racing heart":"tachycardia",
    "tingling":"tingling","pins and needles":"tingling",
    "convulsion":"seizures","fit":"seizures",
    "confused":"confusion","disoriented":"confusion",
    "depressed":"depression","hopeless":"hopelessness",
    "anxious":"anxiety","worried":"anxiety","panic":"panic",
    "appetite":"appetite loss","not hungry":"appetite loss",
    "insomnia":"sleep problems","sleepless":"sleep problems",
    "runny nose":"nasal discharge","blocked nose":"nasal congestion",
    "stuffy":"congestion","congested":"congestion",
}

# Severity modifier weights
SEVERITY_MAP: Dict[str, float] = {
    "severe": 3.0, "excruciating": 3.0, "unbearable": 3.0, "extreme": 3.0,
    "intense": 2.5, "terrible": 2.5, "awful": 2.5, "bad": 2.5, "sharp": 2.5,
    "moderate": 2.0, "significant": 2.0, "considerable": 2.0, "noticeable": 2.0,
    "mild": 1.0, "slight": 1.0, "little": 1.0, "minor": 1.0, "faint": 1.0,
    "occasional": 0.8, "intermittent": 0.8, "sometimes": 0.8,
}

class SpaCyProcessor:
    def __init__(self, model: str = "en_core_web_sm"):
        try:
            self.nlp = spacy.load(model)
        except OSError:
            raise RuntimeError(
                f"spaCy model '{model}' not found.\n"
                f"Run:  python -m spacy download {model}"
            )

    def process(self, text: str) -> Doc:
        return self.nlp(text.lower())

    @staticmethod
    def lemmatize(doc: Doc, stop_words: Set[str]) -> List[str]:
        return [
            token.lemma_ for token in doc
            if not token.is_stop
            and not token.is_punct
            and token.is_alpha
            and len(token.text) > 1
            and token.lemma_ not in stop_words
        ]

    @staticmethod
    def detect_negations_dep(doc: Doc) -> Tuple[Set[str], Set[str]]:
        negated: Set[str] = set()
        negated_heads: Set[int] = set()

        # Collect negated head tokens (verbs/nouns that carry a 'neg' child)
        for token in doc:
            if token.dep_ == "neg":
                negated_heads.add(token.head.i)
                negated.add(token.head.lemma_)

        # Expand negation to the subtree of negated heads
        for token in doc:
            if token.head.i in negated_heads and token.dep_ != "neg":
                if token.pos_ in ("NOUN", "ADJ", "VERB") and not token.is_stop:
                    negated.add(token.lemma_)

        affirmed: Set[str] = {
            t.lemma_ for t in doc
            if not t.is_stop and not t.is_punct and t.is_alpha and len(t.text) > 1
        } - negated

        return affirmed, negated

    @staticmethod
    def extract_entities(doc: Doc) -> List[Dict]:
        return [
            {"text": ent.text, "label": ent.label_, "start": ent.start, "end": ent.end}
            for ent in doc.ents
        ]

    @staticmethod
    def pos_tags(doc: Doc) -> List[Tuple[str, str]]:
        return [(token.text, token.pos_) for token in doc]


class NLPEngine:
    def __init__(self):
        self.spacy = SpaCyProcessor("en_core_web_sm")
        self._stop = STOP_WORDS
        self.medical_context_keywords = {
            "pain", "ache", "sore", "fever", "cough", "tired", "sleep", 
            "eat", "weight", "stomach", "head", "breath", "vision"
        }
        
        # Pre-compute diagnostic vocabulary once for efficiency
        self.all_known_terms = set()
        for cond in CONDITIONS:
            for s in cond.symptoms:
                self.all_known_terms.update(s.lower().replace("_", " ").split())
            for k in cond.keywords:
                self.all_known_terms.update(k.lower().replace("_", " ").split())
        context_signals = {"pain", "fever", "sleep", "weight", "duration", "years", "days", "severe"}
        self.all_known_terms.update(context_signals)
    
    def _extract_medical_signals(self, lemmas: List[str]) -> List[str]:
        return [lemma for lemma in lemmas if lemma in self.all_known_terms]

    def preprocess(self, text: str) -> Dict:
        text = text.lower()
        raw = text
        
        if text.strip().isdigit():
            return {
                "raw": raw,
                "processed_text": "",
                "severity": float(text.strip()),
                "affirmed": set(),
                "negated": set()
            }

        text_norm = self._normalise_synonyms(text)
        
        # Extract features using spaCy pipeline
        doc          = self.spacy.process(text_norm)
        spacy_lemmas = self.spacy.lemmatize(doc, self._stop)
        spacy_affirm, spacy_neg = self.spacy.detect_negations_dep(doc)
        spacy_ents   = self.spacy.extract_entities(doc)
        spacy_pos    = self.spacy.pos_tags(doc)

        combined_lemmas = list(set(spacy_lemmas))
        medical_signals = self._extract_medical_signals(combined_lemmas)
        combined_affirm = spacy_affirm - spacy_neg
        
        processed_text = " ".join(medical_signals if medical_signals else combined_lemmas)

        return {
            "raw":             raw,
            "processed_text":  processed_text,
            "spacy_lemmas":    spacy_lemmas,
            "spacy_entities":  spacy_ents,
            "spacy_pos":       spacy_pos,
            "all_lemmas":      combined_lemmas,
            "affirmed":        combined_affirm,
            "negated":         spacy_neg,
            "severity":        self._extract_severity(text),
            "duration":        self._extract_duration(text),
        }
    
    def get_suggested_category(self, text: str, asked_cats: List[str]) -> str:
        text = text.lower()
        if "severity" not in asked_cats:
            if any(w in text for w in ["pain", "hurt", "ache", "bad", "sharp", "sore"]):
                return "severity"
        
        if "context" not in asked_cats:
            if any(w in text for w in ["eat", "food", "stomach", "tummy", "crave", "weight"]):
                return "context"
        
        if "duration" not in asked_cats:
            if any(w in text for w in ["tired", "sleep", "weak", "exhausted", "long"]):
                return "duration"
        remaining = [c for c in ["duration", "severity", "context", "medications", "associated"] 
                     if c not in asked_cats]
        return random.choice(remaining) if remaining else "associated"

    # Helpers
    @staticmethod
    def _normalise_synonyms(text: str) -> str:
        text_l = text.lower()
        for src, tgt in SYNONYMS.items():
            text_l = re.sub(r"\b" + re.escape(src) + r"\b", tgt, text_l)
        return text_l

    @staticmethod
    def _extract_severity(text: str) -> float:
        text_l = text.lower()
        best = 1.5
        for word, score in SEVERITY_MAP.items():
            if word in text_l:
                best = max(best, score)
        return best

    @staticmethod
    def _extract_duration(text: str) -> Optional[str]:
        patterns = [
            r"(?:for|since|past|last)\s+(\w+\s+(?:hour|day|week|month|year)s?)",
            r"(\d+)\s+(?:hour|day|week|month|year)s?",
            r"(yesterday|today|this\s+morning|this\s+week|few\s+days|couple\s+of\s+days)",
        ]
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                return m.group(0)
        return None

class SklearnEnsemble:
    def __init__(self, conditions: List[Condition]):
        self.conditions = conditions
        self.le = LabelEncoder()

        # 1. Word-level SVM pipeline (with probability calibration)
        self.word_pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                sublinear_tf=True,
                min_df=1,
                max_df=0.95,
                analyzer="word",
            )),
            ("clf", CalibratedClassifierCV(
                LinearSVC(C=1.0, max_iter=3000, class_weight="balanced"),
                cv=3, method="sigmoid",
            )),
        ])

        # 2. Character-level Naive Bayes pipeline (for robust handling of typos/partial matches)
        self.char_tfidf = TfidfVectorizer(
            ngram_range=(3, 5),
            sublinear_tf=True,
            analyzer="char_wb",
            min_df=2,
        )
        self.char_nb = ComplementNB(alpha=0.3)

        # 3. TF-IDF Profile matrix for direct cosine similarity matches
        self.cond_profiles: List[str] = [
            " ".join(c.symptoms + c.keywords + [c.description])
            for c in conditions
        ]
        self._profile_matrix_word = None

    def fit(self, texts: List[str], labels: List[str]):
        y = self.le.fit_transform(labels)

        self.word_pipeline.fit(texts, labels)

        X_char = self.char_tfidf.fit_transform(texts)
        self.char_nb.fit(X_char, y)

        self._profile_matrix_word = (
            self.word_pipeline.named_steps["tfidf"].transform(self.cond_profiles)
        )

    def predict_proba_over_conditions(self, text: str) -> np.ndarray:
        cond_names = [c.name for c in self.conditions]

        word_probs_raw = self.word_pipeline.predict_proba([text])[0]

        # Char NB probabilities
        X_char = self.char_tfidf.transform([text])
        char_log = self.char_nb.predict_log_proba(X_char)[0]
        char_probs_raw = softmax(char_log)

        # Cosine similarity
        X_word = self.word_pipeline.named_steps["tfidf"].transform([text])
        cos_raw = cosine_similarity(X_word, self._profile_matrix_word)[0]

        word_probs = np.zeros(len(cond_names))
        char_probs = np.zeros(len(cond_names))
        for i, label in enumerate(self.le.classes_):
            if label in cond_names:
                idx = cond_names.index(label)
                word_probs[idx] = word_probs_raw[i]
                char_probs[idx] = char_probs_raw[i]

        cos_scores = cos_raw / (cos_raw.sum() + 1e-10)

        final_probs = 0.40 * word_probs + 0.25 * char_probs + 0.35 * cos_scores
        if len(final_probs) != len(self.conditions):
            final_probs = np.resize(final_probs, len(self.conditions))

        return final_probs

class MLEngine:
    #Training and symptom matching 
    URGENCY_BOOST = {"High": 1.4, "Medium": 1.1, "Low": 1.0}

    def __init__(self, conditions: List[Condition]):
        self.conditions = conditions
        self._build_training_data()
        self.sk_clf = SklearnEnsemble(conditions)
        
    def _normalize_training(self, text):
        for k, v in SYNONYMS.items():
            text = re.sub(r"\b" + re.escape(k) + r"\b", v, text.lower())
        return text

    def _build_training_data(self):
        docs, labels = [], []
        rng = np.random.default_rng(42)

        for cond in self.conditions:
            syms = cond.symptoms
            kws  = cond.keywords

            text = " ".join(syms + kws)
            docs.append(self._normalize_training(text))
            labels.append(cond.name)

            for _ in range(12):
                n = max(2, int(rng.integers(2, len(syms) + 1)))
                subset = rng.choice(syms, size=min(n, len(syms)), replace=False).tolist()
                docs.append(self._normalize_training(" ".join(subset)))
                labels.append(cond.name)

            docs.append(self._normalize_training(" ".join(kws)))
            labels.append(cond.name)

            docs.append(self._normalize_training(cond.description))
            labels.append(cond.name)

        self._train_docs   = docs
        self._train_labels = labels

    def train(self):
        self.sk_clf.fit(self._train_docs, self._train_labels)


    def predict(self, user_text: str, nlp_result: Dict, top_n: int = 5) -> List[Dict]:
        combined_text = user_text + " " + nlp_result.get("processed_text", "")

        sk_scores = self.sk_clf.predict_proba_over_conditions(combined_text)

        ensemble = sk_scores
        for i, cond in enumerate(self.conditions):
            match_count = 0
            user_text_l = user_text.lower()
            for sym in cond.symptoms:
                norm_sym = sym.lower().replace("_", " ")
                if norm_sym in user_text_l:
                    match_count += 1
            
            ensemble[i] += 0.2 * match_count
            
        if nlp_result.get("severity", 0) > 7:
            for i, cond in enumerate(self.conditions):
                if cond.urgency == "High":
                    ensemble[i] *= 1.2

        negated = nlp_result.get("negated", set())
        for i in range(min(len(ensemble), len(self.conditions))):
            cond = self.conditions[i]
            profile = " ".join(cond.symptoms + cond.keywords).lower()
            overlap = sum(1 for neg in negated if neg in profile)
            penalty = max(0.1, 1.0 - 0.15 * overlap)
            ensemble[i] *= penalty

        sev_mult = nlp_result.get("severity", 1.5)
        for i in range(min(len(ensemble), len(self.conditions))):
            cond = self.conditions[i]
            urgency_boost = self.URGENCY_BOOST.get(cond.urgency, 1.0)
            ensemble[i] *= ((sev_mult / 1.5) * urgency_boost) ** 0.3

        if len(ensemble) < len(self.conditions):
            ensemble = np.pad(ensemble, (0, len(self.conditions) - len(ensemble)))

        ensemble = ensemble[:len(self.conditions)]

        # Retrieve and format the top 5 results
        top_idxs = np.argsort(ensemble)[::-1][:top_n]
        results = []
        for idx in top_idxs:
            if idx >= len(self.conditions):
                continue
            cond = self.conditions[idx]
            prob = float(ensemble[idx])
            results.append({
                "name":            cond.name,
                "icd":             cond.icd,
                "probability":     prob,
                "probability_pct": f"{min(prob, 1) * 100:.1f}%",
                "severity":        cond.severity,
                "urgency":         cond.urgency,
                "description":     cond.description,
                "recommendations": cond.recommendations,
                "body_system":     cond.body_system,
            })
        return results

# Follow-up Question Engine Bank
FOLLOWUP_BANK: Dict[str, List[str]] = {
    "duration": [
        "How long have you had these symptoms? (e.g. a few hours, 2 days, a week)",
        "When did the symptoms first appear — was the onset sudden or gradual?",
    ],
    "severity": [
        "On a scale of 1–10, how severe are your symptoms right now?",
        "Are the symptoms mild, moderate, or severe — and worsening or improving?",
    ],
    "context": [
        "Have you been in contact with anyone ill, travelled recently, or eaten something unusual?",
        "Did symptoms start after physical exertion, stress, or a specific exposure?",
    ],
    "medications": [
        "Are you currently taking any medications, supplements, or herbal remedies?",
        "Have you been diagnosed with any conditions or had surgery recently?",
    ],
    "associated": [
        "Are there any other symptoms, even if they seem unrelated?",
        "Any recent changes in appetite, weight, sleep, or bathroom habits?",
    ],
}

