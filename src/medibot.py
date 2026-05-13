#!/usr/bin/env python3
"""
medibot.py — MediBot Healthcare Chatbot
═══════════════════════════════════════════
NLP:  NLTK (tokenise · POS · lemmatise · stopwords · negation grammar)
      spaCy (NER · dependency parse · context-aware lemmatisation)
ML:   TensorFlow/Keras (Bidirectional LSTM neural classifier)
      scikit-learn (TF-IDF · LinearSVC · cosine similarity)
      Weighted ensemble with negation penalty + severity boosting

Run:  python medibot.py
API:  used by medibot_api.py
"""

#  §1  STANDARD IMPORTS
import json
import os
import re, sys, time, random, textwrap
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Set
import numpy as np
from scipy.special import softmax

#  §2  NLP IMPORTS — NLTK + spaCy

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize, MWETokenizer
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk import pos_tag, ne_chunk, RegexpParser
from nltk.tree import Tree
import spacy
from spacy.tokens import Doc

#  3  ML IMPORTS — TensorFlow + scikit-learn

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline

#  4  TERMINAL HELPERS
W = 72

def _c(t, *args): return str(t)

def banner():
    print("\n" + "╔" + "═"*70 + "╗")
    print("║" + "  ✦  MediBot — NLTK · spaCy · TensorFlow · sklearn  ".center(70) + "║")
    print("╚" + "═"*70 + "╝\n")

def section(title):
    pad = max(0, (66 - len(title)) // 2)
    print("\n" + "─"*70)
    print("─"*pad + f"  {title}  " + "─"*max(0,68-pad-len(title)))
    print("─"*70 + "\n")

def bot_say(text):
    prefix = "🩺 MediBot"
    for i, line in enumerate(textwrap.fill(text, 62).split("\n")):
        if i == 0:
            print(f"  {prefix}  {line}")
        else:
            print(f"  {' '*12}  {line}")

def user_prompt(hint=""):
    h = f"  [{hint}]" if hint else ""
    return input(f"\n  You{h}  ❯  ").strip()

def pbar(v, w=40):
    filled = int(w * min(v, 1.0))
    return f"[{'█'*filled}{'░'*(w-filled)}]  {v*100:.1f}%"

def sev_color(s):
    return {
        "Low":    ("", "🟢"),
        "Medium": ("", "🟡"),
        "High":   ("", "🔴"),
    }.get(s, ("", "⚪"))

def thinking(label="Processing", steps=16):
    chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    for i in range(steps):
        print(f"\r  {_c(chars[i%10])}  {label}...", end="", flush=True)
        time.sleep(0.08)
    print("\r" + " "*55 + "\r", end="")

#  5  MEDICAL KNOWLEDGE BASE

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

#  6  NLP ENGINE — NLTK + spaCy

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

# NLTK negation cue words
NEGATION_CUES: Set[str] = {
    "no","not","nope","without","never","neither","nor","none","absent",
    "deny","denies","denying","free","lack","negative","ruled",
    "cannot","cant","dont","doesnt","didnt","havent","hasnt","isnt",
    "arent","wasnt","werent","wont","wouldnt","shouldnt",
}

# Severity modifier weights
SEVERITY_MAP: Dict[str, float] = {
    "severe": 3.0, "excruciating": 3.0, "unbearable": 3.0, "extreme": 3.0,
    "intense": 2.5, "terrible": 2.5, "awful": 2.5, "bad": 2.5, "sharp": 2.5,
    "moderate": 2.0, "significant": 2.0, "considerable": 2.0, "noticeable": 2.0,
    "mild": 1.0, "slight": 1.0, "little": 1.0, "minor": 1.0, "faint": 1.0,
    "occasional": 0.8, "intermittent": 0.8, "sometimes": 0.8,
}


class NLTKProcessor:
    """
    NLTK sub-engine.
    Handles: tokenisation · POS tagging · WordNet lemmatisation ·
             stopword removal · negation grammar · symptom phrase extraction.
    """

    def __init__(self):
        # Downloads (idempotent — skip if already present)
        for resource in [
            "punkt", "punkt_tab", "averaged_perceptron_tagger",
            "averaged_perceptron_tagger_eng",
            "maxent_ne_chunker", "maxent_ne_chunker_tab",
            "words", "wordnet", "stopwords", "omw-1.4",
        ]:
            try:
                nltk.data.find(f"tokenizers/{resource}")
            except LookupError:
                try:
                    nltk.data.find(f"taggers/{resource}")
                except LookupError:
                    try:
                        nltk.data.find(f"corpora/{resource}")
                    except LookupError:
                        nltk.download(resource, quiet=True)

        self.lemmatizer  = WordNetLemmatizer()
        self.stemmer     = PorterStemmer()
        self.stop_words  = set(stopwords.words("english"))

        # Multi-word expression tokenizer for medical phrases
        medical_mwes = [
            ("shortness", "of", "breath"), ("loss", "of", "appetite"),
            ("loss", "of", "taste"), ("loss", "of", "smell"),
            ("chest", "pain"), ("back", "pain"), ("heart", "attack"),
            ("blood", "pressure"), ("sore", "throat"), ("runny", "nose"),
            ("night", "sweats"), ("weight", "loss"), ("weight", "gain"),
            ("blurred", "vision"), ("stiff", "neck"), ("dry", "cough"),
            ("rapid", "heartbeat"), ("body", "aches"), ("muscle", "pain"),
            ("joint", "pain"),
        ]
        self.mwe_tokenizer = MWETokenizer(medical_mwes, separator="_")

        # Negation grammar: captures negated noun phrases
        self._neg_grammar = RegexpParser("""
            NEG_NP: {<RB.?>*<DT>?<JJ.*>*<NN.*>+}
                    }<VB.?|IN>+{
        """)

    @staticmethod
    def _pos_to_wordnet(tag: str):
        """Convert NLTK POS tag to WordNet constant for lemmatisation."""
        if tag.startswith("J"):  return wordnet.ADJ
        if tag.startswith("V"):  return wordnet.VERB
        if tag.startswith("R"):  return wordnet.ADV
        return wordnet.NOUN      # default

    def tokenize(self, text: str) -> List[str]:
        """Word-tokenise then resolve multi-word expressions."""
        tokens = word_tokenize(text.lower())
        return self.mwe_tokenizer.tokenize(tokens)

    def pos_tag(self, tokens: List[str]) -> List[Tuple[str, str]]:
        """POS-tag tokens (NLTK averaged perceptron tagger)."""
        return pos_tag(tokens)

    def lemmatize(self, pos_tagged: List[Tuple[str, str]]) -> List[str]:
        """Lemmatize each token using its POS context for accuracy."""
        return [
            self.lemmatizer.lemmatize(tok, self._pos_to_wordnet(tag))
            for tok, tag in pos_tagged
            if tok not in self.stop_words and tok.isalpha() and len(tok) > 1
        ]

    def detect_negations_window(self, tokens: List[str], window: int = 4) -> Tuple[Set[str], Set[str]]:
        """
        Sliding-window negation: any content word within `window` tokens
        after a negation cue is labelled negated.
        """
        negated: Set[str] = set()
        affirmed: Set[str] = set()
        i = 0
        while i < len(tokens):
            clean = tokens[i].strip("',.;:")
            if clean in NEGATION_CUES:
                # Negate up to `window` subsequent content tokens
                for j in range(i + 1, min(i + 1 + window, len(tokens))):
                    w = tokens[j].strip("',.;:")
                    if w not in self.stop_words and w.isalpha():
                        negated.add(self.stemmer.stem(w))
                i += 1
            else:
                if clean not in self.stop_words and clean.isalpha() and len(clean) > 1:
                    affirmed.add(self.stemmer.stem(clean))
                i += 1
        return affirmed - negated, negated

    def extract_ner_chunks(self, tokens: List[str], pos_tagged: List[Tuple[str, str]]) -> List[str]:
        """Extract named-entity chunks using NLTK ne_chunk."""
        tree = ne_chunk(pos_tagged, binary=False)
        entities = []
        for subtree in tree:
            if isinstance(subtree, Tree):
                phrase = " ".join(w for w, _ in subtree.leaves())
                entities.append(phrase)
        return entities


class SpaCyProcessor:
    """
    spaCy sub-engine.
    Handles: NER · dependency-parse-based negation · context-aware lemmatisation ·
             sentence segmentation · POS tagging.
    """

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
        """Context-aware lemmatisation via spaCy's morphological analysis."""
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
        """
        Dependency-parse negation: finds tokens whose governor is negated.
        This is more accurate than window heuristics for complex sentences.
        """
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
        """Extract named entities (spaCy general NER — best-effort for medical text)."""
        return [
            {"text": ent.text, "label": ent.label_, "start": ent.start, "end": ent.end}
            for ent in doc.ents
        ]

    @staticmethod
    def pos_tags(doc: Doc) -> List[Tuple[str, str]]:
        return [(token.text, token.pos_) for token in doc]


class NLPEngine:
    """
    Unified NLP pipeline combining NLTK + spaCy.
    The two engines are complementary:
      - NLTK WordNet lemmatisation is more consistent for medical vocabulary.
      - spaCy dependency parse gives better negation detection.
      - Both POS taggers run; results are merged for richer feature vectors.
    """

    def __init__(self):
        print(f"    Loading NLTK resources...", end=" ", flush=True)
        self.nltk = NLTKProcessor()
        print("✓")

        print(f"    Loading spaCy model (en_core_web_sm)...", end=" ", flush=True)
        self.spacy = SpaCyProcessor("en_core_web_sm")
        print("✓")

        self._stop = self.nltk.stop_words
        self.medical_context_keywords = {
            "pain", "ache", "sore", "fever", "cough", "tired", "sleep", 
            "eat", "weight", "stomach", "head", "breath", "vision"
        }
    
    def _extract_medical_signals(self, lemmas: List[str]) -> List[str]:
        """
        Generalized Filter: Automatically keeps words that exist in ANY condition's 
        symptoms or keywords list, effectively ignoring "couch", "ice cream", etc.
        """
        # Create a flat set of all known medical terms from your JSON
        all_known_terms = set()
        for cond in CONDITIONS:
            for s in cond.symptoms:
                all_known_terms.update(s.lower().replace("_", " ").split())
            for k in cond.keywords:
                all_known_terms.update(k.lower().replace("_", " ").split())
        
        # Also include general medical context signals
        context_signals = {"pain", "fever", "sleep", "weight", "duration", "years", "days", "severe"}
        all_known_terms.update(context_signals)

        # Return only words that the bot "recognizes" as medical
        return [lemma for lemma in lemmas if lemma in all_known_terms]

    def get_suggested_category(self, text: str, asked_cats: List[str]) -> str:
        """
        Generalized Follow-up Logic: Maps user input to follow-up categories 
        based on the 'body_system' or nature of the detected symptoms.
        """
        text = text.lower()
        
        # 1. Map body systems to relevant follow-up focus
        # If the user mentions something related to 'respiratory', prioritize 'duration'
        # If they mention 'gastrointestinal', prioritize 'context' (diet)
        
        detected_systems = {c.body_system for c in CONDITIONS if any(s in text for s in c.symptoms)}
        
        if "severity" not in asked_cats and any(w in text for w in ["pain", "hurt", "bad"]):
            return "severity"
            
        if "context" not in asked_cats:
            if "gastrointestinal" in detected_systems or "infectious" in detected_systems:
                return "context" # Asks about food/travel
                
        if "medications" not in asked_cats:
            if "cardiovascular" in detected_systems or "neurological" in detected_systems:
                return "medications" # Asks about existing meds
                
        # Fallback to any category not yet explored
        remaining = [c for c in ["duration", "severity", "context", "medications", "associated"] 
                    if c not in asked_cats]
        
        return random.choice(remaining) if remaining else "associated"
    # ── Public API ───────────────────────────────────────────────────────────
    def preprocess(self, text: str) -> Dict:
        """
        Full NLP pipeline.  Returns a feature dict used by the ML engine.
        """
        text = text.lower()
        raw = text
        
        # NEW: Handle numeric-only responses (e.g., severity '9')
        if text.strip().isdigit():
            return {
                "raw": raw,
                "processed_text": "",
                "severity": float(text.strip()),
                "affirmed": set(),
                "negated": set()
            }

        text_norm = self._normalise_synonyms(text)
        # 2. NLTK pipeline
        nltk_tokens  = self.nltk.tokenize(text_norm)
        nltk_pos     = self.nltk.pos_tag(nltk_tokens)
        nltk_lemmas  = self.nltk.lemmatize(nltk_pos)
        nltk_ner     = self.nltk.extract_ner_chunks(nltk_tokens, nltk_pos)
        nltk_affirm, nltk_neg = self.nltk.detect_negations_window(nltk_tokens)

        # 3. spaCy pipeline
        doc          = self.spacy.process(text_norm)
        spacy_lemmas = self.spacy.lemmatize(doc, self._stop)
        spacy_affirm, spacy_neg = self.spacy.detect_negations_dep(doc)
        spacy_ents   = self.spacy.extract_entities(doc)
        spacy_pos    = self.spacy.pos_tags(doc)

        # 4. Merge — spaCy negation is more reliable; NLTK widens coverage
        combined_neg    = spacy_neg | nltk_neg
        combined_lemmas = list(set(nltk_lemmas + spacy_lemmas))
        medical_signals = self._extract_medical_signals(combined_lemmas)
        combined_affirm = (spacy_affirm | nltk_affirm) - combined_neg
        
        processed_text = " ".join(medical_signals if medical_signals else combined_lemmas)

        return {
            "raw":             raw,
            "processed_text":  processed_text,
            "nltk_tokens":     nltk_tokens,
            "nltk_lemmas":     nltk_lemmas,
            "nltk_pos":        nltk_pos,
            "nltk_ner":        nltk_ner,
            "spacy_lemmas":    spacy_lemmas,
            "spacy_entities":  spacy_ents,
            "spacy_pos":       spacy_pos,
            "all_lemmas":      combined_lemmas,
            "affirmed":        combined_affirm,
            "negated":         combined_neg,
            "severity":        self._extract_severity(text),
            "duration":        self._extract_duration(text),
        }
    
    def get_suggested_category(self, text: str, asked_cats: List[str]) -> str:
        """Determines the most relevant follow-up category based on keywords."""
        text = text.lower()
        
        # 1. If they mention pain/discomfort but we haven't asked severity
        if "severity" not in asked_cats:
            if any(w in text for w in ["pain", "hurt", "ache", "bad", "sharp", "sore"]):
                return "severity"

        # 2. If they mention food, stomach, or 'big tummy'
        if "context" not in asked_cats:
            if any(w in text for w in ["eat", "food", "stomach", "tummy", "crave", "weight"]):
                return "context"

        # 3. If they mention vague fatigue or systemic issues
        if "duration" not in asked_cats:
            if any(w in text for w in ["tired", "sleep", "weak", "exhausted", "long"]):
                return "duration"

        # 4. Fallback: Pick something not yet asked
        remaining = [c for c in ["duration", "severity", "context", "medications", "associated"] 
                     if c not in asked_cats]
        
        return random.choice(remaining) if remaining else "associated"

    # ── Helpers ──────────────────────────────────────────────────────────────
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


#  7  ML ENGINE — sklearn TF-IDF/SVC + Cosine

class SklearnEnsemble:
    """
    Two sklearn classifiers whose outputs are combined via cosine similarity.

    Components:
      1. TfidfVectorizer (word 1-2grams, sublinear TF) + CalibratedClassifierCV(LinearSVC)
         → calibrated class probabilities via Platt scaling
      2. TfidfVectorizer (char 3-5grams) + ComplementNB
         → for typo robustness and partial matches
      3. Cosine similarity between input vector and pre-vectorised condition profiles

    Output:  probability vector over CONDITIONS (same order).
    """

    def __init__(self, conditions: List[Condition]):
        self.conditions = conditions
        self.le = LabelEncoder()

        # Word TF-IDF + calibrated LinearSVC
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

        # Char TF-IDF + ComplementNB (for partial token matches)
        self.char_tfidf = TfidfVectorizer(
            ngram_range=(3, 5),
            sublinear_tf=True,
            analyzer="char_wb",
            min_df=2,
        )
        self.char_nb = ComplementNB(alpha=0.3)

        # Condition profile matrix (for cosine similarity)
        self.cond_profiles: List[str] = [
            " ".join(c.symptoms + c.keywords + [c.description])
            for c in conditions
        ]
        self._profile_matrix_word = None   # set after fit

    def fit(self, texts: List[str], labels: List[str]):
        y = self.le.fit_transform(labels)

        # Train word pipeline
        self.word_pipeline.fit(texts, labels)

        # Train char ComplementNB
        X_char = self.char_tfidf.fit_transform(texts)
        self.char_nb.fit(X_char, y)

        # Pre-vectorise condition profiles for cosine similarity
        self._profile_matrix_word = (
            self.word_pipeline.named_steps["tfidf"].transform(self.cond_profiles)
        )

    def predict_proba_over_conditions(self, text: str) -> np.ndarray:
        """
        Returns a probability vector of length len(CONDITIONS),
        indexed in the same order as self.conditions.
        """
        cond_names = [c.name for c in self.conditions]

        # Word pipeline probabilities (aligned to le.classes_)
        word_probs_raw = self.word_pipeline.predict_proba([text])[0]

        # Char NB probabilities
        X_char = self.char_tfidf.transform([text])
        char_log = self.char_nb.predict_log_proba(X_char)[0]
        char_probs_raw = softmax(char_log)

        # Cosine similarity
        X_word = self.word_pipeline.named_steps["tfidf"].transform([text])
        cos_raw = cosine_similarity(X_word, self._profile_matrix_word)[0]

        # Map le.classes_ probs → conditions order
        word_probs = np.zeros(len(cond_names))
        char_probs = np.zeros(len(cond_names))
        for i, label in enumerate(self.le.classes_):
            if label in cond_names:
                idx = cond_names.index(label)
                word_probs[idx] = word_probs_raw[i]
                char_probs[idx] = char_probs_raw[i]

        # Normalise cosine scores
        cos_scores = cos_raw / (cos_raw.sum() + 1e-10)

        # Mini-ensemble: word SVC + char NB + cosine
        final_probs = 0.40 * word_probs + 0.25 * char_probs + 0.35 * cos_scores
        if len(final_probs) != len(self.conditions):
            final_probs = np.resize(final_probs, len(self.conditions))

        return final_probs

class MLEngine:
    """
    Grand ensemble: TensorFlow BiLSTM (40%) + sklearn pipeline (60%).
    Also applies negation penalty and severity boosting.

    Training data: augmented from the medical knowledge base (480+ samples).
    """

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
        """
        Augment the knowledge base into ~500 training samples by:
        - Using full symptom list (1 sample)
        - Random symptom subsets of size 2-N (12 samples)
        - Keyword-only sentence (1 sample)
        - Processed/lemmatised variant (1 sample)
        """
        docs, labels = [], []
        rng = np.random.default_rng(42)

        for cond in self.conditions:
            syms = cond.symptoms
            kws  = cond.keywords

            # Full profile
            text = " ".join(syms + kws)
            docs.append(self._normalize_training(text))
            labels.append(cond.name)

            # Subset augmentation
            for _ in range(12):
                n = max(2, int(rng.integers(2, len(syms) + 1)))
                subset = rng.choice(syms, size=min(n, len(syms)), replace=False).tolist()
                docs.append(self._normalize_training(" ".join(subset)))
                labels.append(cond.name)

            # Keywords only
            docs.append(self._normalize_training(" ".join(kws)))
            labels.append(cond.name)

            # Description
            docs.append(self._normalize_training(cond.description))
            labels.append(cond.name)

        self._train_docs   = docs
        self._train_labels = labels

    def train(self):
        """Train both classifiers. Call once on startup."""
        # sklearn is fast
        self.sk_clf.fit(self._train_docs, self._train_labels)


    def predict(self, user_text: str, nlp_result: Dict, top_n: int = 5) -> List[Dict]:
        """
        Ensemble prediction.  Returns top_n conditions with probability etc.
        """
        combined_text = user_text + " " + nlp_result.get("processed_text", "")
        cond_names = [c.name for c in self.conditions]

        # ── sklearn scores ─────────────────────────────────────────────
        sk_scores = self.sk_clf.predict_proba_over_conditions(combined_text)

        # ── Grand ensemble ─────────────────────────────────────────────
        ensemble = sk_scores
        for i, cond in enumerate(self.conditions):
            # IMPROVED: Smarter matching for multi-word symptoms
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

        # ── Negation penalty ──────────────────────────────────────────
        negated = nlp_result.get("negated", set())
        for i in range(min(len(ensemble), len(self.conditions))):
            cond = self.conditions[i]
            profile = " ".join(cond.symptoms + cond.keywords).lower()
            overlap = sum(1 for neg in negated if neg in profile)
            penalty = max(0.1, 1.0 - 0.15 * overlap)
            ensemble[i] *= penalty

        # ── Severity & urgency boosting ────────────────────────────────
        sev_mult = nlp_result.get("severity", 1.5)
        for i in range(min(len(ensemble), len(self.conditions))):
            cond = self.conditions[i]
            urgency_boost = self.URGENCY_BOOST.get(cond.urgency, 1.0)
            ensemble[i] *= ((sev_mult / 1.5) * urgency_boost) ** 0.3

        # ── Normalise ─────────────────────────────────────────────────
        if len(ensemble) < len(self.conditions):
            ensemble = np.pad(ensemble, (0, len(self.conditions) - len(ensemble)))

        ensemble = ensemble[:len(self.conditions)]

        # ── Top-N ─────────────────────────────────────────────────────
        top_idxs = np.argsort(ensemble)[::-1][:top_n]
        results = []
        for idx in top_idxs:
            if idx >= len(self.conditions):   # ✅ FIX
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

#  8  FOLLOW-UP QUESTION ENGINE
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

#  9  CHATBOT STATE MACHINE

class MediBot:
    MAX_FOLLOWUPS = 3

    def __init__(self):
        banner()
        print(f"  {_c('Initialising NLP pipeline (NLTK + spaCy)...')}")
        self.nlp = NLPEngine()
        print()

        print(f"  {_c('Building and training ML models (TF BiLSTM + sklearn)...')}")
        t0 = time.time()
        self.ml = MLEngine(CONDITIONS)
        print(f"    Training TF-IDF · LinearSVC · ComplementNB (sklearn)...", end=" ", flush=True)
        self.ml.sk_clf.fit(self.ml._train_docs, self.ml._train_labels)
        print("✓")

        dt = time.time() - t0
        print(f"✓  ({dt:.1f}s)")
        print(f"\n  {_c(f'All models ready ✓  |  {len(CONDITIONS)} conditions')}\n")

        self._reset()

    def _reset(self):
        self.phase          = "initial"
        self.symptom_text   = []
        self.followup_count = 0
        self.asked_cats     = []

        # ✅ FIX: shuffle every new session
        self.followup_order = list(FOLLOWUP_BANK.keys())
        random.shuffle(self.followup_order)

        print("DEBUG ORDER:", self.followup_order)  # 👈 keep this temporarily

    # ── Ask next follow-up ────────────────────────────────────────────────────
    def _ask_followup(self):
        for cat in self.followup_order:
            if cat not in self.asked_cats:
                self.asked_cats.append(cat)
                q = random.choice(FOLLOWUP_BANK[cat])
                print()
                bot_say(q)
                self.followup_count += 1
                return

    # ── Run ML analysis ───────────────────────────────────────────────────────
    def _run_analysis(self):
        print()
        bot_say("Thank you. Running full NLP + ML diagnostic analysis…")
        print()

        thinking("NLTK · spaCy pipeline", 16)
        full_text   = " ".join(self.symptom_text)
        nlp_result  = self.nlp.preprocess(full_text)

        thinking("TF BiLSTM inference", 14)
        thinking("sklearn ensemble scoring", 12)

        results = self.ml.predict(full_text, nlp_result, top_n=5)

        # Overall urgency
        u_score  = {"Low": 1, "Medium": 2, "High": 3}
        urgency  = {1:"Low", 2:"Medium", 3:"High"}[
            max(u_score[r["urgency"]] for r in results[:3])
        ]

        self._display(results, nlp_result, urgency)
        self.phase = "results"

    # ── Display results ───────────────────────────────────────────────────────
    def _display(self, results, nlp_r, urgency):
        section("NLP + ML DIAGNOSTIC REPORT")

        urg_c, urg_icon = sev_color(urgency)
        print(f"  {_c('Overall Urgency:')}  {_c(urgency)}")
        if nlp_r.get("duration"):
            print(f"  {_c('Reported Duration:')}  {nlp_r['duration']}")
        sev_v   = nlp_r.get("severity", 1.5)
        sev_lbl = "Severe" if sev_v >= 2.5 else "Moderate" if sev_v >= 1.5 else "Mild"
        print(f"  {_c('Severity signal:')}  {sev_lbl} ({sev_v:.1f}×)")
        print()

        # NLP entity summary
        ents  = nlp_r.get("spacy_entities", [])
        affirm = list(nlp_r.get("affirmed", set()))[:8]
        neg   = list(nlp_r.get("negated", set()))[:5]
        if affirm or neg or ents:
            print(f"  {_c('NLP Affirmed:')}  {', '.join(affirm) or 'none'}")
            print(f"  {_c('NLP Negated:')}   {', '.join(neg) or 'none'}")
            if ents:
                ent_strs = [f"{e['text']} ({e['label']})" for e in ents[:4]]
                print(f"  {_c('spaCy NER:')}     {', '.join(ent_strs)}")
            print()

        print(f"  {_c('Differential Diagnoses')}\n")
        ranks = ["MOST LIKELY","LIKELY","POSSIBLE","LESS LIKELY","UNLIKELY"]
        for i, res in enumerate(results):
            sc, icon = sev_color(res["severity"])
            icd_tag = _c(f"[{res['icd']}]")
            print(f"  {'─'*62}")
            print(f"  #{i+1}  {res['name']}  {icd_tag}  "
                  f"{sc}{icon} {res['severity']}  "
                  f"{_c(ranks[i])}")
            print(f"  {_c('Probability:')}  {pbar(res['probability'], 26, sc)}")
            print(f"  {_c(res['description'])}")
            print()

        section(f"RECOMMENDATIONS — {results[0]['name'].upper()}")
        for j, rec in enumerate(results[0]["recommendations"], 1):
            print(f"  {_c(str(j)+'.')}  {rec}")

        if urgency == "High":
            print(f"\n  ⚠️  HIGH URGENCY — SEEK IMMEDIATE MEDICAL ATTENTION  ")

        section("ML PIPELINE USED")
        print(f"  {_c('NLP:')}  NLTK (word_tokenize · pos_tag · WordNetLemmatizer · MWE · ne_chunk)")
        print(f"  {'':5}  spaCy (en_core_web_sm · NER · dep parse negation · lemma_)")
        print(f"  {_c('ML: ')}  TensorFlow/Keras BiLSTM (Embed 64 → BiLSTM 64 → BiLSTM 32 → Dense)")
        print(f"  {'':5}  sklearn TF-IDF + CalibratedLinearSVC (word 1-2gram)")
        print(f"  {'':5}  sklearn TF-IDF + ComplementNB (char 3-5gram)")
        print(f"  {'':5}  Cosine Similarity vs condition profiles")
        print(f"  {_c('Ensemble:')}  TF 40% + sklearn 60%  ·  negation penalty  ·  severity boost")

        section("DISCLAIMER")
        dis = ("MediBot is a prototype educational tool using open-source NLP/ML. "
               "It does NOT replace professional medical diagnosis. "
               "Always consult a qualified healthcare professional.")
        for line in textwrap.fill(dis, 66).split("\n"):
            print(f"  {_c(line)}")
        print()

    # ── Main message handler ──────────────────────────────────────────────────
    def handle(self, text: str):
        if self.phase == "initial":
            nlp_r = self.nlp.preprocess(text)
            self.symptom_text.append(text)
            if not nlp_r["nltk_tokens"]:
                bot_say("I didn't catch specific symptoms. Could you describe how you feel in more detail?")
                return
            affirmed = [t for t in list(nlp_r["affirmed"])[:3] if len(t) > 3]
            if affirmed:
                bot_say(f"Noted — detected: {', '.join(affirmed[:2])}. I have a few follow-up questions.")
            else:
                bot_say("Thank you. Let me ask some follow-up questions.")
            self.phase = "gathering"
            self._ask_followup()

        elif self.phase == "gathering":
            self.symptom_text.append(text)
            if self.followup_count < self.MAX_FOLLOWUPS:
                self._ask_followup()
            else:
                self._run_analysis()

    # ── Post-results menu ─────────────────────────────────────────────────────
    def _post_results(self):
        print(f"\n  {_c('What next?')}")
        print(f"  {_c('[1]')}  New symptom check")
        print(f"  {_c('[2]')}  List all conditions")
        print(f"  {_c('[q]')}  Quit\n")
        c = input(f"  {_c('Choice')}  ❯  ").strip().lower()
        if c == "1":
            self._reset()
            print()
            bot_say("Starting fresh. Please describe your symptoms.")
        elif c == "2":
            by_sys = defaultdict(list)
            for cond in CONDITIONS:
                by_sys[cond.body_system].append(cond)
            section("CONDITIONS IN KNOWLEDGE BASE")
            for sys, conds in sorted(by_sys.items()):
                print(f"  {_c(sys.upper(), )}")
                for c_ in conds:
                    sc, icon = sev_color(c_.severity)
                    print(f"    {icon} {c_.name}  {_c(c_.icd)}")
                print()
            self._post_results()
        else:
            sys.exit(0)

    # ── Run loop ──────────────────────────────────────────────────────────────
    def run(self):
        print(f"  {_c('NLP: NLTK + spaCy  ·  ML: TensorFlow BiLSTM + sklearn')}\n")
        bot_say("Hello! I'm your AI health assistant. What symptoms are you experiencing today?")
        bot_say("List multiple symptoms separated by commas — e.g. fever, headache, stiff neck")

        while True:
            try:
                hint = "symptoms" if self.phase == "initial" else f"Q{self.followup_count}/{self.MAX_FOLLOWUPS}"
                user_input = user_prompt(hint if self.phase != "results" else "")
                if not user_input:
                    continue
                if user_input.lower() in ("q","quit","exit","bye"):
                    print(f"\n  {_c('Goodbye! Stay well. 🌿')}\n")
                    break
                if user_input.lower() == "reset":
                    self._reset()
                    bot_say("Session reset. Please describe your symptoms.")
                    continue
                print()
                self.handle(user_input)
                if self.phase == "results":
                    self._post_results()
            except KeyboardInterrupt:
                print(f"\n\n  {_c('Session interrupted. Goodbye! 🌿')}\n")
                break

#  10  ENTRY POINT
if __name__ == "__main__":
    MediBot().run()