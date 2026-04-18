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

import tensorflow as tf # type: ignore
from tensorflow import keras # type: ignore
from tensorflow.keras import layers # type: ignore
from tensorflow.keras.preprocessing.text import Tokenizer as KerasTokenizer # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore

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

CONDITIONS: List[Condition] = [
    Condition(
        "Common Cold","J00",
        ["runny nose","nasal congestion","sneezing","sore throat","mild cough",
         "low grade fever","mild headache","watery eyes","fatigue"],
        ["cold","runny","congestion","sniffle","sneezing","stuffy"],
        "Low","Low",
        "Viral upper respiratory infection, usually rhinovirus. Self-limiting in 7–10 days.",
        ["Rest and stay hydrated","Saline nasal spray for congestion",
         "OTC antihistamines or decongestants","Throat lozenges or honey for sore throat"],
        "respiratory"
    ),
    Condition(
        "Influenza (Flu)","J11",
        ["high fever","severe body aches","muscle pain","fatigue","chills",
         "dry cough","headache","sore throat","sweating","weakness"],
        ["flu","influenza","aches","chills","fatigue","sweating","shivering"],
        "Medium","Medium",
        "Highly contagious respiratory illness from influenza A/B viruses.",
        ["Antiviral medication within 48 h of onset","Rest and fluids",
         "Acetaminophen or ibuprofen for fever","Isolate until 24 h after fever breaks"],
        "respiratory"
    ),
    Condition(
        "COVID-19","U07.1",
        ["fever","dry cough","fatigue","loss of taste","loss of smell",
         "shortness of breath","body aches","headache","sore throat","diarrhea"],
        ["covid","coronavirus","taste","smell","oxygen","isolate","breathless"],
        "Medium","High",
        "Respiratory illness from SARS-CoV-2, ranging mild to critical.",
        ["Isolate immediately","Monitor SpO₂ — seek ER if below 94%",
         "Rest and hydration","Follow local public-health guidance"],
        "respiratory"
    ),
    Condition(
        "Pneumonia","J18",
        ["productive cough","chest pain","high fever","chills","shortness of breath",
         "rapid breathing","fatigue","sweating","confusion","phlegm"],
        ["pneumonia","lung","chest","phlegm","productive","consolidation"],
        "High","High",
        "Lung infection — bacterial, viral, or fungal — causing air-sac inflammation.",
        ["Seek immediate medical evaluation","Antibiotics for bacterial pneumonia",
         "Hospital admission likely for high-risk patients","Supplemental oxygen if SpO₂ low"],
        "respiratory"
    ),
    Condition(
        "Asthma","J45",
        ["wheezing","shortness of breath","chest tightness","cough",
         "nocturnal cough","exercise breathlessness"],
        ["asthma","wheeze","inhaler","tightness","breathless","trigger","pollen"],
        "Medium","Medium",
        "Chronic inflammatory airway disease with recurrent wheeze and breathlessness.",
        ["Use rescue inhaler (salbutamol) as prescribed","Identify and avoid triggers",
         "Ensure preventer inhaler compliance","ER if inhaler gives no relief"],
        "respiratory"
    ),
    Condition(
        "Bronchitis","J40",
        ["persistent cough","mucus","chest congestion","mild fever","fatigue",
         "sore throat","wheezing","chest discomfort"],
        ["bronchitis","bronchial","mucus","phlegm","productive cough","smoker"],
        "Low","Low",
        "Bronchial tube inflammation, usually viral following a cold.",
        ["Rest and increase fluids","Steam inhalation to loosen mucus",
         "Avoid smoking","Doctor if cough persists > 3 weeks"],
        "respiratory"
    ),
    Condition(
        "Allergic Rhinitis","J30",
        ["sneezing","runny nose","itchy eyes","nasal congestion","watery eyes",
         "itchy nose","post-nasal drip"],
        ["allergy","pollen","hay fever","itchy","watery","dust","seasonal"],
        "Low","Low",
        "Immune hypersensitivity to airborne allergens causing nasal inflammation.",
        ["Antihistamines (loratadine/cetirizine)","Intranasal corticosteroid spray",
         "Avoid allergen when possible","Allergy testing for long-term management"],
        "respiratory"
    ),
    Condition(
        "Migraine","G43",
        ["severe headache","throbbing pain","nausea","vomiting",
         "light sensitivity","aura","visual disturbances","one-sided headache"],
        ["migraine","throbbing","aura","photophobia","one side","nausea"],
        "Medium","Medium",
        "Neurological disorder with recurrent severe headaches and sensory disturbances.",
        ["Rest in dark quiet room","Triptans or NSAIDs early in attack",
         "Identify triggers via headache diary","Preventive medication if frequent"],
        "neurological"
    ),
    Condition(
        "Tension Headache","G44.2",
        ["dull headache","pressure around head","tight band feeling",
         "neck tension","shoulder tension"],
        ["tension","pressure","tight","band","stress","bilateral","dull"],
        "Low","Low",
        "Most common headache type caused by muscle tension, stress, or poor posture.",
        ["OTC analgesics (ibuprofen, paracetamol)","Warm compress on neck/shoulders",
         "Relaxation techniques","Improve posture and take screen breaks"],
        "neurological"
    ),
    Condition(
        "Sinusitis","J01",
        ["facial pain","nasal congestion","thick nasal discharge","post-nasal drip",
         "reduced smell","headache","facial pressure","toothache"],
        ["sinus","sinusitis","facial pressure","thick discharge","congested"],
        "Low","Low",
        "Sinus inflammation usually following a cold or allergic rhinitis.",
        ["Saline nasal irrigation","Steam inhalation","OTC decongestants short-term",
         "Doctor if symptoms > 10 days"],
        "respiratory"
    ),
    Condition(
        "Gastroenteritis","A09",
        ["nausea","vomiting","diarrhea","abdominal cramps","stomach pain",
         "low grade fever","loss of appetite","dehydration"],
        ["gastro","stomach bug","vomit","diarrhea","cramps","food poisoning"],
        "Medium","Medium",
        "Stomach/intestine inflammation from viral or bacterial infection.",
        ["Oral rehydration salts (ORS)","BRAT diet: banana rice applesauce toast",
         "Avoid dairy and spicy foods","ER if unable to keep fluids down > 24 h"],
        "gastrointestinal"
    ),
    Condition(
        "Acid Reflux / GERD","K21",
        ["heartburn","chest burning","acid taste","regurgitation",
         "difficulty swallowing","bloating","belching","chronic cough"],
        ["heartburn","reflux","gerd","acid","burning","regurgitate","belch"],
        "Low","Low",
        "Stomach acid flows back into oesophagus, causing irritation.",
        ["Avoid spicy/fatty/caffeinated foods","Eat smaller meals",
         "Elevate head of bed","OTC antacids or H2 blockers"],
        "gastrointestinal"
    ),
    Condition(
        "Appendicitis","K37",
        ["severe right lower abdominal pain","nausea","vomiting","fever",
         "loss of appetite","rebound tenderness","guarding"],
        ["appendix","appendicitis","right lower","rebound","guarding","sharp"],
        "High","High",
        "Acute appendix inflammation — surgical emergency if untreated.",
        ["SEEK EMERGENCY CARE IMMEDIATELY","Do NOT take laxatives or antacids",
         "Surgery (appendectomy) is standard treatment"],
        "gastrointestinal"
    ),
    Condition(
        "Urinary Tract Infection (UTI)","N39.0",
        ["burning urination","frequent urination","urgency","cloudy urine",
         "strong smell","pelvic pain","lower abdominal pain","blood in urine"],
        ["uti","urinary","bladder","burning","frequent","cloudy","pelvic"],
        "Medium","Medium",
        "Bacterial infection of the urinary tract, most commonly the bladder.",
        ["Doctor for antibiotic prescription","Drink plenty of water",
         "Avoid caffeine and alcohol","Urgent care if fever or back pain develops"],
        "urinary"
    ),
    Condition(
        "Kidney Stones","N20",
        ["severe flank pain","back pain","radiating groin pain",
         "blood in urine","nausea","vomiting","painful urination"],
        ["kidney stone","flank","groin","radiate","severe","wave","colic"],
        "High","High",
        "Hard mineral deposits in kidneys causing intense pain as they pass.",
        ["Drink 2–3 L water daily","NSAIDs or prescription pain relief",
         "Doctor immediately if fever develops","Lithotripsy for large stones"],
        "urinary"
    ),
    Condition(
        "Hypertension","I10",
        ["headache","dizziness","blurred vision","pounding chest",
         "shortness of breath","nosebleeds","fatigue","often no symptoms"],
        ["blood pressure","hypertension","high bp","dizziness","nosebleed"],
        "Medium","Medium",
        "Chronically elevated blood pressure increasing cardiovascular risk. Often silent.",
        ["Monitor blood pressure regularly","DASH diet (low sodium)","Regular exercise",
         "Limit alcohol and quit smoking","Take antihypertensives as prescribed"],
        "cardiovascular"
    ),
    Condition(
        "Heart Attack (MI)","I21",
        ["crushing chest pain","chest pressure","left arm pain","jaw pain",
         "shortness of breath","sweating","nausea","pale","anxiety"],
        ["heart attack","crushing","chest","left arm","jaw","sudden","sweat"],
        "High","High",
        "Coronary artery blockage causing cardiac muscle death.",
        ["CALL 108/112 IMMEDIATELY","Chew 300 mg aspirin if not allergic",
         "Lie down and rest","Do NOT drive yourself"],
        "cardiovascular"
    ),
    Condition(
        "Stroke","I63",
        ["sudden facial drooping","arm weakness","speech difficulty",
         "sudden severe headache","vision changes","loss of balance"],
        ["stroke","facial droop","arm weakness","speech","balance","sudden"],
        "High","High",
        "Brain blood-vessel blockage or rupture — use FAST assessment.",
        ["CALL EMERGENCY SERVICES IMMEDIATELY — time = brain cells",
         "FAST: Face, Arm, Speech, Time to call","Note symptom onset time",
         "Do NOT give food or water"],
        "neurological"
    ),
    Condition(
        "Anxiety Disorder","F41",
        ["excessive worry","restlessness","muscle tension","irritability",
         "sleep problems","difficulty concentrating","rapid heartbeat","sweating"],
        ["anxiety","anxious","worry","nervous","panic","fear","tension","restless"],
        "Medium","Medium",
        "Persistent excessive worry that interferes with daily activities.",
        ["Deep breathing (4-7-8 technique)","Regular aerobic exercise",
         "Limit caffeine","Consider CBT therapy","Medication if severe"],
        "mental_health"
    ),
    Condition(
        "Depression","F32",
        ["persistent sadness","loss of interest","fatigue","worthlessness",
         "sleep disturbance","appetite changes","difficulty concentrating","hopelessness"],
        ["depressed","depression","sad","hopeless","worthless","interest","withdraw"],
        "Medium","High",
        "Persistent depressive disorder affecting mood, behaviour, and quality of life.",
        ["Seek professional mental health support","Talk therapy (CBT/IPT) is effective",
         "Regular light exercise","Maintain social connections"],
        "mental_health"
    ),
    Condition(
        "Panic Attack","F41.0",
        ["sudden intense fear","pounding heart","sweating","trembling",
         "shortness of breath","chest pain","nausea","dizziness","fear of dying"],
        ["panic attack","sudden fear","pounding","tremble","shake","control","dying"],
        "Medium","Medium",
        "Sudden intense fear with physical symptoms; peaks within 10 minutes.",
        ["Grounding: 5-4-3-2-1 senses","Diaphragmatic breathing",
         "Reassure yourself it will pass","Avoid caffeine","CBT to prevent recurrence"],
        "mental_health"
    ),
    Condition(
        "Anemia","D50",
        ["fatigue","pale skin","shortness of breath","cold hands","dizziness",
         "rapid heartbeat","brittle nails","headache","poor concentration"],
        ["anemia","anaemia","pale","iron","dizzy","cold","nail","weakness"],
        "Medium","Medium",
        "Insufficient red blood cells or haemoglobin reducing oxygen delivery.",
        ["Blood test to identify anemia type","Iron-rich diet (red meat, spinach, lentils)",
         "Vitamin C with iron supplements","Treat underlying cause"],
        "haematological"
    ),
    Condition(
        "Hypothyroidism","E03",
        ["fatigue","weight gain","cold intolerance","constipation","dry skin",
         "hair loss","depression","slow heart rate","puffy face"],
        ["thyroid","hypothyroid","weight gain","cold intolerance","hair loss","sluggish"],
        "Medium","Low",
        "Underactive thyroid producing insufficient hormones. Common in women.",
        ["TSH blood test for diagnosis","Daily levothyroxine as prescribed",
         "Take on empty stomach 30-60 min before food","Monitor thyroid every 6–12 months"],
        "endocrine"
    ),
    Condition(
        "Hyperthyroidism","E05",
        ["weight loss","rapid heartbeat","anxiety","heat intolerance","sweating",
         "tremor","increased appetite","diarrhea","bulging eyes"],
        ["hyperthyroid","overactive thyroid","weight loss","heat","tremor","graves"],
        "Medium","Medium",
        "Overactive thyroid producing excess hormones; can cause cardiac complications.",
        ["Endocrinologist referral essential","Antithyroid medications",
         "Beta-blockers for heart rate control","Radioactive iodine therapy"],
        "endocrine"
    ),
    Condition(
        "Type 2 Diabetes","E11",
        ["increased thirst","frequent urination","blurred vision","slow healing",
         "fatigue","numbness in feet","frequent infections","increased hunger"],
        ["diabetes","blood sugar","glucose","thirst","insulin","foot numb"],
        "Medium","Medium",
        "Chronic insulin resistance causing elevated blood glucose.",
        ["Measure fasting blood glucose and HbA1c","Low-GI diet","150+ min exercise/week",
         "Lose 5–10% body weight if overweight","Metformin is first-line medication"],
        "endocrine"
    ),
    Condition(
        "Dehydration","E86",
        ["dark urine","dizziness","dry mouth","decreased urination","headache",
         "fatigue","muscle cramps","confusion","rapid heartbeat"],
        ["dehydrated","thirsty","dry mouth","dark urine","cramps","electrolyte"],
        "Low","Medium",
        "Insufficient body fluid impairing all physiological functions.",
        ["Drink water or ORS immediately","Avoid caffeine and alcohol",
         "Sports drinks for electrolyte replacement","IV fluids if unable to keep down"],
        "general"
    ),
    Condition(
        "Meningitis","G03",
        ["severe headache","stiff neck","high fever","photophobia","phonophobia",
         "nausea","vomiting","rash","confusion","seizures"],
        ["meningitis","stiff neck","light sensitive","petechiae","rash","severe"],
        "High","High",
        "Meninges inflammation — can be bacterial (emergency) or viral.",
        ["MEDICAL EMERGENCY — call 108/112 immediately",
         "Non-blanching rash = meningococcal septicaemia emergency",
         "IV antibiotics must start immediately"],
        "neurological"
    ),
    Condition(
        "Food Poisoning","A05",
        ["nausea","vomiting","diarrhea","stomach cramps","fever",
         "weakness","onset hours after eating"],
        ["food poisoning","contaminated","ate something","restaurant","vomit"],
        "Medium","Medium",
        "Illness from contaminated food/water; usually self-limiting.",
        ["ORS to maintain hydration","BRAT diet when able",
         "Seek care if symptoms > 48 h or fever > 38.5°C"],
        "gastrointestinal"
    ),
    Condition(
        "Eczema","L20",
        ["itchy skin","dry skin","red patches","oozing blisters","thickened skin",
         "skin inflammation","rash","flaking"],
        ["eczema","atopic","dermatitis","itch","dry skin","rash","flare"],
        "Low","Low",
        "Chronic inflammatory skin condition causing dry, itchy, inflamed patches.",
        ["Moisturise frequently with fragrance-free emollient","Avoid triggers",
         "Topical corticosteroids for flares","Antihistamines at night for itch"],
        "dermatological"
    ),
    Condition(
        "Gout","M10",
        ["sudden intense joint pain","joint swelling","redness","warmth",
         "tenderness","big toe pain","limited movement"],
        ["gout","uric acid","big toe","swollen","red","warm","tender","purine"],
        "Medium","Medium",
        "Inflammatory arthritis from urate crystal deposition in joints.",
        ["NSAIDs or colchicine for acute attack","Ice pack on joint",
         "Avoid alcohol and high-purine foods","Allopurinol for long-term control"],
        "musculoskeletal"
    ),
    Condition(
        "Lower Back Pain","M54.5",
        ["lower back pain","stiffness","limited movement","muscle spasm",
         "radiating leg pain","sciatica"],
        ["back pain","lumbar","spine","stiff","sciatica","disc","posture"],
        "Low","Low",
        "Very common musculoskeletal complaint; usually mechanical and self-limiting.",
        ["Stay active — bed rest is not recommended","Heat/ice packs",
         "OTC NSAIDs or paracetamol","Core-strengthening physiotherapy"],
        "musculoskeletal"
    ),
    Condition(
        "Conjunctivitis (Pink Eye)","H10",
        ["red eyes","eye discharge","itchy eyes","watery eyes",
         "swollen eyelids","eye crusting","blurred vision","light sensitivity"],
        ["conjunctivitis","pink eye","red eye","discharge","crusty","itchy eye"],
        "Low","Low",
        "Inflammation of the conjunctiva — bacterial, viral, or allergic.",
        ["Warm compress for crusty discharge","Antibiotic drops for bacterial type",
         "Avoid touching eyes","Wash hands frequently to prevent spread"],
        "ophthalmological"
    ),
]

#  6  NLP ENGINE — NLTK + spaCy

# Medical synonym normalisation map
SYNONYMS: Dict[str, str] = {
    "tummy":"stomach","belly":"stomach","gut":"stomach",
    "throat":"throat","pharynx":"throat",
    "chest":"chest","thorax":"chest",
    "cardiac":"heart","cardio":"heart",
    "gastric":"stomach","abdominal":"stomach abdomen",
    "lumbar":"back","spine":"back","spinal":"back",
    "articular":"joint",
    "myalgia":"muscle pain",
    "breath":"breathing","respiratory":"breathing",
    "urinate":"urination","pee":"urination","urine":"urination",
    "puke":"vomiting","nausea":"nausea",
    "dizzy":"dizziness","lightheaded":"dizziness","vertigo":"dizziness",
    "tired":"fatigue","exhausted":"fatigue","lethargic":"fatigue",
    "temp":"fever","feverish":"fever","pyrexia":"fever",
    "itch":"itching","itchy":"itching",
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

    # ── Public API ───────────────────────────────────────────────────────────
    def preprocess(self, text: str) -> Dict:
        """
        Full NLP pipeline.  Returns a feature dict used by the ML engine.
        """
        raw = text

        # 1. Synonym normalisation (before tokenising)
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
        combined_affirm = (spacy_affirm | nltk_affirm) - combined_neg

        # 5. Scalar features
        severity = self._extract_severity(text)
        duration = self._extract_duration(text)

        # 6. Processed text for vectorisers
        processed_text = " ".join(combined_lemmas)

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
            "severity":        severity,
            "duration":        duration,
        }

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


#  7  ML ENGINE — TensorFlow Keras BiLSTM + sklearn TF-IDF/SVC + Cosine

class TFBiLSTMClassifier:
    """
    Bidirectional LSTM text classifier built with TensorFlow / Keras.

    Architecture:
      Embedding (vocab × 64, mask_zero=True)
      → Bidirectional LSTM (64, return_sequences=True, dropout=0.2)
      → Bidirectional LSTM (32, dropout=0.2)
      → Dense (64, relu)
      → BatchNormalization
      → Dropout (0.4)
      → Dense (num_classes, softmax)

    Training: sparse categorical cross-entropy · Adam (lr 0.001)
              EarlyStopping (patience=5) · ReduceLROnPlateau
    """

    VOCAB_SIZE = 3500
    MAX_LEN    = 70
    EMBED_DIM  = 64

    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.keras_tokenizer = KerasTokenizer(
            num_words=self.VOCAB_SIZE,
            oov_token="<OOV>",
            filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
        )
        self.label_encoder = LabelEncoder()
        self.model: Optional[keras.Sequential] = None
        self._fitted = False

    def _build_model(self) -> keras.Sequential:
        model = keras.Sequential([
            layers.Embedding(
                input_dim=self.VOCAB_SIZE,
                output_dim=self.EMBED_DIM,
                input_length=self.MAX_LEN,
                mask_zero=True,
            ),
            layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.1)),
            layers.Bidirectional(layers.LSTM(32, dropout=0.2)),
            layers.Dense(64, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(self.num_classes, activation="softmax"),
        ], name="medibot_bilstm")

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        seqs = self.keras_tokenizer.texts_to_sequences(texts)
        return pad_sequences(seqs, maxlen=self.MAX_LEN, padding="post", truncating="post")

    def fit(self, texts: List[str], labels: List[str], epochs: int = 40, batch_size: int = 16):
        self.keras_tokenizer.fit_on_texts(texts)
        X = self._encode_texts(texts)
        y = self.label_encoder.fit_transform(labels)

        self.model = self._build_model()
        self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=0,
            callbacks=[
                EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
                ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5),
            ],
        )
        self._fitted = True

    def predict_proba(self, text: str) -> np.ndarray:
        """Returns probability array aligned to self.label_encoder.classes_."""
        if not self._fitted:
            raise RuntimeError("Model not trained yet. Call fit() first.")
        X = self._encode_texts([text])
        return self.model.predict(X, verbose=0)[0]   # shape: (num_classes,)

    @property
    def classes_(self):
        return self.label_encoder.classes_


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
        return 0.40 * word_probs + 0.25 * char_probs + 0.35 * cos_scores


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

        n_classes = len(conditions)

        self.tf_clf = TFBiLSTMClassifier(n_classes)
        self.sk_clf = SklearnEnsemble(conditions)

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
            docs.append(" ".join(syms + kws))
            labels.append(cond.name)

            # Subset augmentation
            for _ in range(12):
                n = max(2, int(rng.integers(2, len(syms) + 1)))
                subset = rng.choice(syms, size=min(n, len(syms)), replace=False).tolist()
                docs.append(" ".join(subset))
                labels.append(cond.name)

            # Keywords only
            docs.append(" ".join(kws))
            labels.append(cond.name)

            # Description
            docs.append(cond.description)
            labels.append(cond.name)

        self._train_docs   = docs
        self._train_labels = labels

    def train(self):
        """Train both classifiers. Call once on startup."""
        # sklearn is fast
        self.sk_clf.fit(self._train_docs, self._train_labels)

        # TF BiLSTM
        self.tf_clf.fit(
            self._train_docs,
            self._train_labels,
            epochs=40,
            batch_size=16,
        )

    def predict(self, user_text: str, nlp_result: Dict, top_n: int = 5) -> List[Dict]:
        """
        Ensemble prediction.  Returns top_n conditions with probability etc.
        """
        combined_text = user_text + " " + nlp_result.get("processed_text", "")
        cond_names = [c.name for c in self.conditions]

        # ── sklearn scores ─────────────────────────────────────────────
        sk_scores = self.sk_clf.predict_proba_over_conditions(combined_text)

        # ── TF BiLSTM scores ───────────────────────────────────────────
        tf_raw  = self.tf_clf.predict_proba(combined_text)          # len = num_classes
        tf_scores = np.zeros(len(cond_names))
        for i, label in enumerate(self.tf_clf.classes_):
            if label in cond_names:
                tf_scores[cond_names.index(label)] = tf_raw[i]

        # ── Grand ensemble ─────────────────────────────────────────────
        ensemble = 0.40 * tf_scores + 0.60 * sk_scores

        # ── Negation penalty ──────────────────────────────────────────
        negated = nlp_result.get("negated", set())
        for i, cond in enumerate(self.conditions):
            profile = " ".join(cond.symptoms + cond.keywords).lower()
            overlap = sum(1 for neg in negated if neg in profile)
            penalty = max(0.1, 1.0 - 0.15 * overlap)
            ensemble[i] *= penalty

        # ── Severity & urgency boosting ────────────────────────────────
        sev_mult = nlp_result.get("severity", 1.5)
        for i, cond in enumerate(self.conditions):
            urgency_boost = self.URGENCY_BOOST[cond.urgency]
            ensemble[i] *= ((sev_mult / 1.5) * urgency_boost) ** 0.3

        # ── Normalise ─────────────────────────────────────────────────
        total = ensemble.sum()
        if total > 1e-10:
            ensemble /= total

        # ── Top-N ─────────────────────────────────────────────────────
        top_idxs = np.argsort(ensemble)[::-1][:top_n]
        results = []
        for idx in top_idxs:
            cond = self.conditions[idx]
            prob = float(ensemble[idx])
            results.append({
                "name":            cond.name,
                "icd":             cond.icd,
                "probability":     prob,
                "probability_pct": f"{prob * 100:.1f}%",
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

        print(f"    Training Bidirectional LSTM (TensorFlow/Keras)...", end=" ", flush=True)
        self.ml.tf_clf.fit(
            self.ml._train_docs,
            self.ml._train_labels,
            epochs=40,
            batch_size=16,
        )
        dt = time.time() - t0
        print(f"✓  ({dt:.1f}s)")
        print(f"\n  {_c(f'All models ready ✓  |  {len(CONDITIONS)} conditions')}\n")

        self._reset()

    def _reset(self):
        self.phase          = "initial"
        self.symptom_text   = []
        self.followup_count = 0
        self.asked_cats     = []

    # ── Ask next follow-up ────────────────────────────────────────────────────
    def _ask_followup(self):
        order = ["duration","severity","context","medications","associated"]
        for cat in order:
            if cat not in self.asked_cats:
                self.asked_cats.append(cat)
                q = random.choice(FOLLOWUP_BANK[cat])
                print()
                bot_say(q)
                self.followup_count += 1
                return
        # all asked — pick random
        cat = random.choice(order)
        bot_say(random.choice(FOLLOWUP_BANK[cat]))
        self.followup_count += 1

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
                print(f"\n\n  {_c('Session interrupted. Goodbye! 🌿', Fore.CYAN)}\n")
                break

#  10  ENTRY POINT
if __name__ == "__main__":
    MediBot().run()