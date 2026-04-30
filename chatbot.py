"""
chatbot.py  —  NeuroScan Pro
=====================================================================
NeuroBot: Smart clinical EEG assistant.

LLM BACKEND (INTERNET REQUIRED):
  Use Groq's free tier  →  llama-3.3-70b-versatile
  Get a free key at: https://console.groq.com  (no credit card needed)
  Enter it in the chatbot sidebar OR set env var GROQ_API_KEY.

FALLBACK (WHEN NO INTERNET, IT'LL STILL WORK):
  If no API key is set, a high-quality semantic keyword engine handles
  all questions. It understands paraphrasing via synonym expansion —
  not just exact keyword matching.
=====================================================================
"""
from __future__ import annotations
import os, re
from typing import Optional

_GROQ_URL   = "https://api.groq.com/openai/v1/chat/completions"
_GROQ_MODEL = "llama-3.3-70b-versatile"

# ══════════════════════════════════════════════════════════════════
#  SYSTEM PROMPT
# ══════════════════════════════════════════════════════════════════
_SYS_BASE = """
You are NeuroBot, the clinical AI assistant embedded in NeuroScan Pro — a research-grade EEG epilepsy detection system built as a Final Year Project.

ROLE: Help users understand EEG results, seizure types, EEG features, medications, and next steps.

TONE: Professional, warm, concise. Never alarmist. Always remind users this is AI screening, not clinical diagnosis.

FORMAT RULES:
- Use **bold** for clinical terms and key values
- Use bullet points for lists
- Use markdown tables for comparisons — always
- Use headers (##, ###) to structure long answers
- Keep conversational replies to 2–3 sentences; clinical answers use full structure

CLINICAL KNOWLEDGE:

EEG BANDS: delta 0.5–4 Hz / theta 4–8 Hz / alpha 8–13 Hz / beta 13–30 Hz / gamma 30–45 Hz

THE 4 CLASSES:
  NORMAL:       alpha dominant (8–13 Hz), highest entropy, lowest delta, low cross-corr, <0.1 spikes/s
  FOCAL (G40.1): theta dominant (4–8 Hz), LOWEST cross-corr (one region), spikes >7/s, reduced entropy
  ABSENCE (G40.3): 3 Hz SWD, highest delta, LOWEST entropy, highest kurtosis, highest cross-corr, lowest ZCR. Children 4–14.
  TONIC-ATONIC (G40.5/G40.8): HIGHEST ZCR (>120/s), highest amplitude & wavelet energy, highest Hjorth complexity, bilateral

KEY DIFFERENTIATORS:
  Focal vs Absence: Focal is LOCALISED (lowest cross-corr); Absence is BILATERAL (highest cross-corr)
  Absence vs Tonic: Absence LOW ZCR (3 Hz); Tonic HIGHEST ZCR (15–25 Hz fast oscillation)
  Normal vs all: Normal has HIGHEST entropy; seizures reduce entropy

MEDICATIONS:
  Ethosuximide → absence only (first-line)
  Carbamazepine/Oxcarbazepine → focal (first-line; AVOID in absence — worsens it)
  Valproate → broad-spectrum (absence, tonic-atonic, generalised)
  Lamotrigine → focal + generalised; safe in pregnancy
  Levetiracetam → broad add-on, well-tolerated
  Rufinamide → tonic-atonic in Lennox-Gastaut syndrome

AI SYSTEM:
  5-model ensemble: RandomForest (primary gold standard), LogisticRegression, SVM (RBF), KNN (k=21 baseline), CNN-1D (NumPy)
  Split: 70% train / 15% validation / 15% test + 5-fold cross-validation
  Target accuracy: 82–88% on 4-class EEG classification
  46 clinical features per recording

RULES:
- Never say "you definitely have epilepsy" — say "patterns consistent with..."
- Always state this is AI screening, not a clinical diagnosis
- When patient result is available in context, reference it specifically
- For off-topic questions, politely redirect to EEG/epilepsy topics
""".strip()


def _build_context(prediction: Optional[dict]) -> str:
    if not prediction:
        return "PATIENT CONTEXT: No EEG result loaded yet."
    stype  = prediction.get("seizure_type", "Unknown")
    conf   = prediction.get("confidence", 0) * 100
    icd10  = prediction.get("icd10", "—")
    is_epi = prediction.get("is_epileptic", False)
    desc   = prediction.get("description", "")
    domf   = prediction.get("dom_freq")
    bp     = prediction.get("band_power") or {}
    si     = prediction.get("spike_info") or {}
    proba  = prediction.get("class_probabilities") or {}

    lines = [
        "PATIENT CONTEXT (current EEG result):",
        f"  Result      : {stype}",
        f"  ICD-10      : {icd10}",
        f"  Confidence  : {conf:.0f}%",
        f"  Epileptic   : {'Yes' if is_epi else 'No'}",
    ]
    if desc:
        lines.append(f"  Description : {desc}")
    if domf and domf > 0:
        lines.append(f"  Dominant Hz : {domf:.1f} Hz")
    if bp:
        bps = ", ".join(f"{b}={bp.get(b,{}).get('rel',0)*100:.0f}%" for b in ("delta","theta","alpha","beta","gamma"))
        lines.append(f"  Band powers : {bps}")
    if si:
        n = si.get("n_spikes", 0)
        r = si.get("spike_rate_per_s", 0)
        lines.append(f"  Spikes      : {n} ({r:.2f}/s)" if n else "  Spikes      : None")
    if proba:
        lines.append("  Class probs : " + " | ".join(f"{k}: {v*100:.0f}%" for k,v in proba.items()))
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════
#  LLM CALL  —  Groq free tier
# ══════════════════════════════════════════════════════════════════
def _call_groq(messages: list[dict], api_key: str) -> Optional[str]:
    try:
        import requests
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        body    = {"model": _GROQ_MODEL, "messages": messages,
                   "max_tokens": 900, "temperature": 0.35, "stream": False}
        r = requests.post(_GROQ_URL, headers=headers, json=body, timeout=25)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        code = getattr(getattr(e, "response", None), "status_code", None)
        if code == 401:
            return "🔑 **Invalid Groq API key.** Check your key at [console.groq.com](https://console.groq.com)."
        if code == 429:
            return "⏳ **Rate limit reached.** Wait a few seconds and try again."
        return None   # fall through to keyword engine


# ══════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════
def neurobot_respond(
    user_message: str,
    history:      list[dict],
    prediction:   Optional[dict],
    api_key:      str = "",
) -> str:
    api_key = (api_key or os.environ.get("GROQ_API_KEY", "")).strip()

    if api_key:
        ctx    = _build_context(prediction)
        system = f"{_SYS_BASE}\n\n{ctx}"
        msgs: list[dict] = [{"role": "system", "content": system}]
        for m in history[-10:]:
            msgs.append({"role": "user" if m["role"]=="user" else "assistant",
                         "content": m["text"]})
        msgs.append({"role": "user", "content": user_message})
        result = _call_groq(msgs, api_key)
        if result:
            return result

    return _keyword_engine(user_message, prediction)


# ══════════════════════════════════════════════════════════════════
#  SEMANTIC KEYWORD ENGINE  —  fallback (no API needed)
# ══════════════════════════════════════════════════════════════════
_SYNS: dict[str, str] = {
    # greetings
    "hello":"greet","hi":"greet","hey":"greet","good morning":"greet",
    "good afternoon":"greet","good evening":"greet","howdy":"greet",
    "what can you do":"greet","who are you":"greet","start":"greet",
    # result
    "result":"result","my result":"result","diagnosis":"result","detected":"result",
    "what does it mean":"result","explain my result":"result","what did it find":"result",
    "what did it say":"result","what did it show":"result","prediction":"result",
    "what is wrong with me":"result","interpret":"result","output":"result",
    "what was found":"result","analysis result":"result","my eeg result":"result",
    # comparison
    "compare":"compare","comparison":"compare","difference":"compare",
    "differences":"compare","vs":"compare","versus":"compare","between":"compare",
    "distinguish":"compare","contrast":"compare","how are they different":"compare",
    "how do they differ":"compare","all types":"compare","each type":"compare",
    "list all":"compare","explain all":"compare","all seizure types":"compare",
    "which is better":"compare","what are the types":"compare",
    # absence
    "absence":"absence","petit mal":"absence","3 hz":"absence","3hz":"absence",
    "spike wave":"absence","spike-wave":"absence","swd":"absence",
    "staring spell":"absence","blank stare":"absence","childhood epilepsy":"absence",
    "childhood seizure":"absence","staring seizure":"absence",
    # focal
    "focal":"focal","partial seizure":"focal","temporal lobe":"focal",
    "frontal lobe":"focal","localised":"focal","localized":"focal",
    "one hemisphere":"focal","one region":"focal","one side brain":"focal",
    "partial epilepsy":"focal","focal epilepsy":"focal",
    # tonic
    "tonic":"tonic","atonic":"tonic","drop attack":"tonic","grand mal":"tonic",
    "stiffening":"tonic","tonic-atonic":"tonic","tonic clonic":"tonic",
    "drop seizure":"tonic","convulsion":"tonic","muscle stiff":"tonic",
    "falling seizure":"tonic","lennox":"tonic","lennox gastaut":"tonic",
    # normal
    "normal eeg":"normal","healthy brain":"normal","no seizure":"normal",
    "no epilepsy":"normal","alpha rhythm":"normal","normal brain":"normal",
    "not epileptic":"normal","no epileptiform":"normal",
    # medication
    "medication":"meds","medicine":"meds","drug":"meds","tablet":"meds",
    "pill":"meds","prescription":"meds","treatment":"meds","therapy":"meds",
    "valproate":"meds","carbamazepine":"meds","lamotrigine":"meds",
    "levetiracetam":"meds","ethosuximide":"meds","phenytoin":"meds",
    "rufinamide":"meds","anticonvulsant":"meds","anti-seizure":"meds",
    "antiepileptic":"meds","aed":"meds","oxcarbazepine":"meds",
    "what to take":"meds","which medicine":"meds","which medication":"meds",
    # next steps
    "next step":"next","what should i do":"next","what to do":"next",
    "what now":"next","recommend":"next","advice":"next","suggest":"next",
    "see a doctor":"next","neurologist":"next","hospital":"next",
    "after diagnosis":"next","after result":"next","action":"next",
    "guidance":"next","consult":"next","what do i do":"next",
    "should i be worried":"next","is it serious":"next",
    # band power
    "band power":"bands","delta":"bands","theta":"bands","alpha":"bands",
    "beta":"bands","gamma":"bands","frequency band":"bands","spectrum":"bands",
    "power spectrum":"bands","spectral power":"bands","frequency":"bands",
    "band":"bands","hz":"bands",
    # entropy
    "entropy":"entropy","sample entropy":"entropy","spectral entropy":"entropy",
    "permutation entropy":"entropy","complexity":"entropy","randomness":"entropy",
    "irregularity":"entropy","chaotic":"entropy","how complex":"entropy",
    "signal complexity":"entropy","predictable":"entropy",
    # spikes
    "spike":"spikes","interictal":"spikes","epileptiform":"spikes",
    "sharp wave":"spikes","discharge":"spikes","spike rate":"spikes",
    "how many spikes":"spikes","transient":"spikes","spike count":"spikes",
    # hjorth
    "hjorth":"hjorth","mobility":"hjorth","hjorth complexity":"hjorth",
    "hjorth activity":"hjorth","hjorth parameters":"hjorth","signal morphology":"hjorth",
    # ai model
    "how does it work":"ai","ai model":"ai","machine learning":"ai",
    "random forest":"ai","svm":"ai","cnn":"ai","knn":"ai",
    "logistic regression":"ai","ensemble":"ai","algorithm":"ai",
    "neural network":"ai","classifier":"ai","how was it trained":"ai",
    "deep learning":"ai","artificial intelligence":"ai","how accurate is it":"ai",
    "which model":"ai","what models":"ai","5 models":"ai",
    # confidence
    "confidence":"conf","how sure":"conf","accurate":"conf","accuracy":"conf",
    "reliable":"conf","certainty":"conf","probability":"conf","trust":"conf",
    "how confident":"conf","how reliable":"conf","percentage":"conf",
    "what does confidence mean":"conf",
    # epilepsy definition
    "what is epilepsy":"epilepsy","define epilepsy":"epilepsy",
    "epilepsy mean":"epilepsy","about epilepsy":"epilepsy",
    "explain epilepsy":"epilepsy","epilepsy":"epilepsy","what causes epilepsy":"epilepsy",
    # eeg definition
    "what is eeg":"eeg_def","eeg mean":"eeg_def","how does eeg work":"eeg_def",
    "electroencephalogram":"eeg_def","eeg machine":"eeg_def","eeg test":"eeg_def",
    "brain wave":"eeg_def","brainwave":"eeg_def","electroencephalography":"eeg_def",
    "what is an eeg":"eeg_def","explain eeg":"eeg_def",
    # icd10
    "icd":"icd","icd-10":"icd","icd10":"icd","diagnostic code":"icd",
    "classification code":"icd","disease code":"icd","medical code":"icd",
    # causes/triggers
    "cause":"cause","trigger":"cause","why seizure":"cause","risk factor":"cause",
    "what causes":"cause","provoke":"cause","photosensitive":"cause",
    "sleep deprivation":"cause","stress":"cause","what triggers":"cause",
    # training data / model performance
    "training data":"train","train split":"train","validation":"train",
    "test set":"train","cross validation":"train","k fold":"train",
    "how was it trained":"train","data split":"train","overfitting":"train",
    "underfitting":"train","f1 score":"train","precision":"train","recall":"train",
}


def _tokenise(text: str) -> set[str]:
    t = text.lower().strip()
    found: set[str] = set()
    # Multi-word phrases first (longest match priority)
    for phrase in sorted(_SYNS, key=len, reverse=True):
        if phrase in t:
            found.add(_SYNS[phrase])
            t = t.replace(phrase, " ")
    # Single words
    for w in re.sub(r"[^\w\s]", " ", t).split():
        if w in _SYNS:
            found.add(_SYNS[w])
        else:
            found.add(w)
    return found


def _has(tokens: set, *concepts: str) -> bool:
    return any(c in tokens for c in concepts)


def _keyword_engine(question: str, prediction: Optional[dict]) -> str:
    tokens = _tokenise(question)
    p      = prediction or {}
    has_r  = bool(p) and p.get("seizure_type") not in (None, "Unknown")

    def ctx() -> str:
        if not has_r:
            return ""
        s = p.get("seizure_type", "?")
        c = p.get("confidence", 0) * 100
        i = p.get("icd10", "—")
        return f"\n\n---\n📋 *Your current result: **{s}** — {c:.0f}% confidence | ICD-10: {i}*"

    # Priority order
    if _has(tokens, "greet"):
        return _r_greeting()

    if _has(tokens, "compare") or sum(_has(tokens, t) for t in ("absence","focal","tonic","normal")) >= 2:
        return _r_comparison(ctx())

    if _has(tokens, "result") and any(w in question.lower() for w in ("my","what","explain","mean","show","tell")):
        return _r_result(p, has_r)

    if _has(tokens, "absence") and not _has(tokens, "focal","tonic"):
        return _r_absence(ctx())

    if _has(tokens, "focal") and not _has(tokens, "absence","tonic"):
        return _r_focal(ctx())

    if _has(tokens, "tonic") and not _has(tokens, "absence","focal"):
        return _r_tonic(ctx())

    if _has(tokens, "normal") and not _has(tokens, "absence","focal","tonic"):
        return _r_normal(ctx())

    if _has(tokens, "meds"):
        return _r_medication(p, has_r)

    if _has(tokens, "next"):
        return _r_next_steps(p, has_r)

    if _has(tokens, "bands"):
        return _r_bands(p, has_r)

    if _has(tokens, "entropy"):
        return _r_entropy(ctx())

    if _has(tokens, "spikes"):
        return _r_spikes(p, has_r)

    if _has(tokens, "hjorth"):
        return _r_hjorth(ctx())

    if _has(tokens, "ai"):
        return _r_ai()

    if _has(tokens, "conf"):
        return _r_confidence(p, has_r)

    if _has(tokens, "epilepsy"):
        return _r_epilepsy()

    if _has(tokens, "eeg_def"):
        return _r_eeg_def()

    if _has(tokens, "icd"):
        return _r_icd(ctx())

    if _has(tokens, "cause"):
        return _r_causes()

    if _has(tokens, "train"):
        return _r_training()

    return _r_fallback(ctx())


# ══════════════════════════════════════════════════════════════════
#  ANSWER TEMPLATES
# ══════════════════════════════════════════════════════════════════

def _r_greeting() -> str:
    return (
        "👋 **Hi! I'm NeuroBot**, the clinical AI assistant for NeuroScan Pro.\n\n"
        "Here's what I can help you with:\n\n"
        "| Topic | Example |\n"
        "|---|---|\n"
        "| 🔬 Your EEG result | *\"What does my result mean?\"* |\n"
        "| 🧠 Seizure types | *\"What is an absence seizure?\"* |\n"
        "| ⚖️ Comparisons | *\"Compare focal vs absence\"* |\n"
        "| 💊 Medications | *\"What treats focal seizures?\"* |\n"
        "| 📋 Next steps | *\"What should I do after my diagnosis?\"* |\n"
        "| 🌊 EEG features | *\"What is entropy?\"*, *\"Explain band power\"* |\n"
        "| 🤖 AI model | *\"How does the AI work?\"* |\n\n"
        "Just ask naturally — I understand full sentences, not just keywords!"
    )


def _r_comparison(ctx: str) -> str:
    return (
        "## ⚖️ All 4 EEG Classes — Complete Comparison\n\n"
        "### 🟢 Normal EEG\n"
        "Healthy waking brain. **Alpha rhythm (8–13 Hz)** is dominant. "
        "Has the **highest entropy** (complex, unpredictable), "
        "**lowest delta power**, low cross-channel correlation, and almost no spikes (<0.1/s). "
        "No epileptiform patterns.\n\n"
        "### 🟠 Focal Seizure — ICD-10: G40.1\n"
        "Starts in **one brain region**. **Theta dominant (4–8 Hz)**. "
        "**Lowest cross-correlation** of all types — only the ictal zone is firing. "
        "Elevated interictal spikes (>7/s). Reduced entropy. "
        "Symptoms vary by lobe: motor jerking, déjà vu, visual aura.\n\n"
        "### 🔴 Absence Seizure — ICD-10: G40.3\n"
        "Defined by **3 Hz spike-wave discharges**. **Highest delta power** (3 Hz = delta). "
        "**Lowest entropy** (most periodic EEG signal in existence). **Highest kurtosis**. "
        "**Lowest ZCR** (slow wave). **Highest bilateral cross-correlation** (both hemispheres in sync). "
        "Brief staring spells, no convulsions. Mainly children aged 4–14.\n\n"
        "### 🟣 Tonic-Atonic Seizure — ICD-10: G40.5 / G40.8\n"
        "**Highest ZCR** (rapid tonic oscillation, 15–25 Hz). "
        "**Highest amplitude and wavelet energy** (broadband power). "
        "**Highest Hjorth complexity**. Bilateral spread → high cross-correlation. "
        "Tonic = sudden stiffening; Atonic = drop attack (high injury risk).\n\n"
        "---\n"
        "### 📊 Quick Reference Table\n\n"
        "| Feature | 🟢 Normal | 🟠 Focal | 🔴 Absence | 🟣 Tonic-Atonic |\n"
        "|---|---|---|---|---|\n"
        "| Dominant Hz | 8–13 (alpha) | 4–8 (theta) | 3 (SWD) | 15–25 (fast) |\n"
        "| Entropy | ⬆ Highest | Medium | ⬇ Lowest | Low |\n"
        "| Delta power | ⬇ Lowest | Medium | ⬆ Highest | High |\n"
        "| Cross-corr | Low | ⬇ Lowest | ⬆ Highest | High |\n"
        "| ZCR | Medium | Medium | ⬇ Lowest | ⬆ Highest |\n"
        "| Amplitude | Low | Medium | High | ⬆ Highest |\n"
        "| Kurtosis | Normal | Medium | ⬆ Highest | High |\n"
        "| Convulsions | None | Focal only | None | ✅ Stiffening/fall |\n"
        "| Bilateral | — | ❌ No | ✅ Yes | ✅ Yes |\n"
        "| ICD-10 | — | G40.1 | G40.3 | G40.5/G40.8 |\n"
        + ctx
    )


def _r_result(p: dict, has_r: bool) -> str:
    if not has_r:
        return (
            "⚠️ **No EEG result loaded yet.**\n\n"
            "To get your result:\n"
            "1. Go to **🏥 Dashboard** and train the models (or use synthetic data)\n"
            "2. Go to **📁 Upload & Analyse** and upload your EEG file\n"
            "3. Click **Run Full Epilepsy Analysis**\n"
            "4. Come back here — I'll explain exactly what your result means!"
        )

    stype  = p.get("seizure_type", "Unknown")
    conf   = p.get("confidence", 0) * 100
    icd10  = p.get("icd10", "—")
    is_epi = p.get("is_epileptic", False)
    desc   = p.get("description", "")
    bp     = p.get("band_power") or {}
    si     = p.get("spike_info") or {}

    if not is_epi:
        return (
            "## ✅ Your Result: Normal EEG\n\n"
            "The AI found **no epileptic activity** in your recording.\n\n"
            "**What this means:**\n"
            "- **Alpha rhythm (8–13 Hz)** is dominant — the signature of a healthy waking brain\n"
            "- **No spike-wave discharges** or epileptiform patterns detected\n"
            "- **High spectral entropy** — complex, irregular, healthy brain dynamics\n"
            "- **Low cross-channel correlation** — regions oscillate independently\n\n"
            f"**AI Confidence:** {conf:.0f}%\n\n"
            "> ⚠️ A normal result does **not** rule out epilepsy. "
            "Seizures may not occur during the recording window. "
            "Please confirm with a qualified neurologist."
        )

    band_line = ""
    if bp:
        dom = max(("delta","theta","alpha","beta","gamma"), key=lambda b: bp.get(b,{}).get("rel",0))
        rel = bp.get(dom,{}).get("rel",0) * 100
        band_line = f"\n- **Dominant band:** {dom.capitalize()} ({rel:.0f}% of total power)"

    spike_line = ""
    if si and si.get("n_spikes",0) > 0:
        spike_line = f"\n- **Spikes:** {si['n_spikes']} detected ({si['spike_rate_per_s']:.2f}/s)"

    return (
        f"## 🔴 Your Result: {stype} Seizure Detected\n\n"
        f"{desc}\n\n"
        f"**Key findings from your EEG:**"
        f"{band_line}{spike_line}\n\n"
        f"**AI Confidence:** {conf:.0f}%&nbsp;&nbsp;|&nbsp;&nbsp;**ICD-10:** {icd10}\n\n"
        "> ⚠️ This is AI-assisted screening — **not a clinical diagnosis**. "
        "Please consult a **neurologist** for confirmation and treatment planning."
    )


def _r_absence(ctx: str) -> str:
    return (
        "## 🔴 Absence Seizure — ICD-10: G40.3\n\n"
        "**What it is:** Brief (5–30 sec) generalised non-convulsive seizures caused by "
        "rhythmic **3 Hz spike-wave discharges (SWD)** across both hemispheres.\n\n"
        "**EEG signature:**\n"
        "- 3 Hz symmetric SWD — the defining feature\n"
        "- **Highest delta power** (3 Hz is in the delta band)\n"
        "- **Lowest entropy** of all seizure types — extremely periodic pattern\n"
        "- **Highest EEG kurtosis** — very sharp spike peaks\n"
        "- **Highest bilateral cross-correlation** — both hemispheres fire in sync\n"
        "- **Lowest ZCR** — slow 3 Hz oscillation\n\n"
        "**Symptoms:**\n"
        "- Sudden blank stare — patient 'switches off' mid-sentence\n"
        "- No convulsions, no falling, no post-ictal confusion\n"
        "- Immediate return to normal when it ends\n"
        "- Can occur 10–100× per day, often unnoticed by the patient\n\n"
        "**Who it affects:** Children aged 4–14. Often outgrown by adulthood.\n\n"
        "**Treatment:** **Ethosuximide** (first-line), valproate, lamotrigine.\n\n"
        "> ⚠️ **Carbamazepine is contraindicated** — it worsens absence seizures."
        + ctx
    )


def _r_focal(ctx: str) -> str:
    return (
        "## 🟠 Focal (Partial) Seizure — ICD-10: G40.1\n\n"
        "**What it is:** Seizures originating in one localised brain region. "
        "Symptoms depend entirely on which lobe is involved.\n\n"
        "**EEG signature:**\n"
        "- **Theta dominant (4–8 Hz)** in one hemisphere or lobe\n"
        "- **Lowest cross-channel correlation** — only the ictal zone is affected\n"
        "- Elevated interictal spike rate (>7/s) in the focal zone\n"
        "- **Reduced entropy** — more ordered than normal EEG\n"
        "- Asymmetric amplitude across channels\n\n"
        "**Symptoms by lobe:**\n\n"
        "| Lobe | Typical symptoms |\n"
        "|---|---|\n"
        "| Motor cortex | Rhythmic jerking of arm, leg, or face |\n"
        "| Temporal | Déjà vu, strange smells/tastes, fear, automatisms |\n"
        "| Occipital | Visual hallucinations, flashing lights |\n"
        "| Frontal | Bizarre movements, vocalisation (often nocturnal) |\n\n"
        "**Treatment:** **Carbamazepine** or oxcarbazepine (first-line), levetiracetam. "
        "Surgical resection is curative in ~70% of drug-resistant focal cases."
        + ctx
    )


def _r_tonic(ctx: str) -> str:
    return (
        "## 🟣 Tonic-Atonic Seizure — ICD-10: G40.5 / G40.8\n\n"
        "**What it is:** Generalised seizures involving sudden changes in muscle tone.\n\n"
        "**EEG signature:**\n"
        "- **Highest ZCR** (rapid 15–25 Hz tonic oscillation) — primary discriminator\n"
        "- **Highest signal amplitude and wavelet energy** — broadband power\n"
        "- **Highest Hjorth complexity** — most morphologically complex EEG waveform\n"
        "- **High bilateral cross-correlation** — generalised, both hemispheres\n"
        "- Sudden voltage attenuation during atonic phase\n\n"
        "**Clinical phases:**\n"
        "- **Tonic:** Sudden sustained muscle stiffening — arms raise, back arches\n"
        "- **Atonic:** Sudden loss of all muscle tone → drop attack (high injury risk)\n\n"
        "**Treatment:** **Valproate** (first-line), lamotrigine, rufinamide. "
        "VNS or corpus callosotomy for refractory cases.\n\n"
        "> Often associated with **Lennox-Gastaut syndrome**. "
        "A protective helmet is often recommended due to drop attacks."
        + ctx
    )


def _r_normal(ctx: str) -> str:
    return (
        "## 🟢 Normal EEG\n\n"
        "A normal EEG reflects the healthy resting-state activity of the waking brain:\n\n"
        "- **Alpha rhythm (8–13 Hz)** is dominant — the hallmark of relaxed wakefulness\n"
        "- **Highest spectral entropy** — complex, unpredictable, non-periodic activity\n"
        "- **Very low delta and theta power** — slow waves are abnormal in a waking adult\n"
        "- **Low cross-channel correlation** — each region oscillates independently\n"
        "- **Interictal spike rate < 0.1/s** — essentially no epileptiform activity\n"
        "- **High Lyapunov exponent** — chaotic, non-periodic brain dynamics\n\n"
        "> ⚠️ A normal EEG does **not** rule out epilepsy. "
        "Seizures may not occur during the recording window. "
        "A prolonged or sleep EEG may be needed if symptoms persist."
        + ctx
    )


def _r_medication(p: dict, has_r: bool) -> str:
    result_note = ""
    if has_r:
        stype = p.get("seizure_type","")
        specific = {
            "Focal":        "**Carbamazepine** or oxcarbazepine (first-line), levetiracetam. Consider surgery if drug-resistant.",
            "Absence":      "**Ethosuximide** (first-line for pure absence), valproate, lamotrigine. **Avoid carbamazepine** — it worsens absence.",
            "Tonic-Atonic": "**Valproate** (first-line), lamotrigine, rufinamide. Consider VNS or callosotomy for refractory cases.",
        }
        if stype in specific:
            result_note = f"\n\n---\n📋 **Recommended for your result ({stype}):** {specific[stype]}"

    return (
        "## 💊 Anti-Seizure Medications\n\n"
        "| Medication | Best for | Key note |\n"
        "|---|---|---|\n"
        "| **Ethosuximide** | Absence only | First-line; does not cover other types |\n"
        "| **Carbamazepine** | Focal | First-line; **avoid in absence** |\n"
        "| **Oxcarbazepine** | Focal | Fewer side effects than carbamazepine |\n"
        "| **Valproate** | Broad-spectrum | Absence, tonic-atonic, generalised |\n"
        "| **Lamotrigine** | Focal + generalised | Safe in pregnancy; titrate slowly |\n"
        "| **Levetiracetam** | Add-on, broad | Well-tolerated, few drug interactions |\n"
        "| **Rufinamide** | Tonic-atonic | For Lennox-Gastaut syndrome |\n"
        "| **Phenytoin** | Focal / tonic-clonic | Older agent; narrow therapeutic window |\n"
        + result_note
        + "\n\n> ⚠️ Never start, stop, or change medication without your neurologist's guidance."
    )


def _r_next_steps(p: dict, has_r: bool) -> str:
    if not has_r:
        return (
            "## 📋 Getting Started\n\n"
            "No EEG result loaded yet. Here's what to do:\n\n"
            "1. Go to **🏥 Dashboard** → train models or use synthetic data\n"
            "2. Go to **📁 Upload & Analyse** → upload your EEG file (CSV, XLSX, EDF, or image)\n"
            "3. Click **Run Full Epilepsy Analysis**\n"
            "4. Come back here — I'll give personalised next-step guidance!"
        )

    stype  = p.get("seizure_type","")
    conf   = p.get("confidence",0)*100
    icd10  = p.get("icd10","—")
    is_epi = p.get("is_epileptic",False)

    if not is_epi:
        return (
            "## ✅ Next Steps — Normal EEG\n\n"
            "1. **Share results with your neurologist** for clinical confirmation\n"
            "2. **Keep a symptom diary** — note any unusual episodes or blank spells\n"
            "3. **Request a prolonged or sleep EEG** if symptoms continue\n"
            "4. **No treatment needed** based on this screening — follow up with your doctor\n\n"
            f"> AI Confidence: {conf:.0f}% &nbsp;|&nbsp; "
            "> ⚠️ This is AI screening — always confirm with a neurologist."
        )

    steps = {
        "Focal": (
            "1. **Consult a neurologist** — request a clinical EEG and brain MRI\n"
            "2. **First-line medication:** Carbamazepine, oxcarbazepine, or levetiracetam\n"
            "3. **Do not drive** until seizure-free for the legally required period\n"
            "4. **Discuss surgery** if medication fails — resection is curative in ~70%\n"
            "5. Avoid triggers: sleep deprivation, alcohol, stress"
        ),
        "Absence": (
            "1. **Consult a paediatric neurologist** (most common in children)\n"
            "2. **First-line medication:** Ethosuximide or valproate\n"
            "3. **Avoid triggers:** hyperventilation, flickering lights, sleep deprivation\n"
            "4. **Prognosis is good** — many children outgrow absence epilepsy by their teens\n"
            "5. Monitor school performance — frequent staring spells affect concentration"
        ),
        "Tonic-Atonic": (
            "1. **Seek urgent neurological evaluation** — high injury risk from drop attacks\n"
            "2. **Safety first:** wear a protective helmet, pad sharp furniture corners\n"
            "3. **Medications:** valproate, lamotrigine, rufinamide\n"
            "4. **Discuss VNS** or corpus callosotomy for refractory cases\n"
            "5. **Do not drive or swim alone** — drop attacks occur without warning"
        ),
    }.get(stype,
        "1. Consult a neurologist for full clinical EEG evaluation\n"
        "2. Begin appropriate anti-seizure medication as prescribed\n"
        "3. Avoid seizure triggers and unsafe activities (driving, swimming alone)"
    )

    return (
        f"## ⚠️ Next Steps — {stype} Seizure Detected\n\n"
        f"{steps}\n\n"
        f"> AI Confidence: {conf:.0f}% &nbsp;|&nbsp; ICD-10: {icd10}\n\n"
        "> ⚠️ This is AI-assisted screening. Always confirm with a qualified neurologist."
    )


def _r_bands(p: dict, has_r: bool) -> str:
    base = (
        "## 🌊 EEG Frequency Bands\n\n"
        "| Band | Range | What it signals |\n"
        "|---|---|---|\n"
        "| **Delta** | 0.5–4 Hz | Deep sleep; ⬆ in **absence** (3 Hz SWD) & tonic-atonic |\n"
        "| **Theta** | 4–8 Hz | Drowsiness; ⬆ dominant in **focal** seizures |\n"
        "| **Alpha** | 8–13 Hz | Relaxed wakefulness — dominant in **normal** brain |\n"
        "| **Beta** | 13–30 Hz | Active thinking, alertness, benzodiazepine effect |\n"
        "| **Gamma** | 30–45 Hz | High-level processing; ⬆ in **tonic** burst phase |\n"
    )
    if has_r:
        bp = p.get("band_power") or {}
        if bp:
            base += "\n\n---\n📋 **Your recorded band powers:**\n\n"
            base += "| Band | Power | Bar |\n|---|---|---|\n"
            for b in ("delta","theta","alpha","beta","gamma"):
                rel = bp.get(b,{}).get("rel",0)*100
                bar = "█" * int(rel/4) + "░" * max(0, 25-int(rel/4))
                base += f"| {b.capitalize()} | {rel:.1f}% | `{bar}` |\n"
    return base


def _r_entropy(ctx: str) -> str:
    return (
        "## 📐 Entropy in EEG\n\n"
        "**Entropy** measures how complex and unpredictable the EEG signal is. "
        "The rule: **high entropy = healthy brain; low entropy = seizure activity.**\n\n"
        "| Level | Meaning | Seen in |\n"
        "|---|---|---|\n"
        "| ⬆ Highest | Complex, random, unpredictable | 🟢 Normal healthy brain |\n"
        "| Medium | Partially ordered | 🟠 Focal or mild abnormality |\n"
        "| Low | Periodic, ordered | 🟣 Tonic-Atonic |\n"
        "| ⬇ Lowest | Extremely repetitive (3 Hz) | 🔴 Absence seizure |\n\n"
        "**Types of entropy used in NeuroScan Pro:**\n"
        "- **Sample entropy** — regularity of short time-series patterns\n"
        "- **Spectral entropy** — spread of energy across frequency bands\n"
        "- **Permutation entropy** — ordinal symbol patterns in the signal\n"
        "- **Lempel-Ziv complexity** — compressibility of the binary signal"
        + ctx
    )


def _r_spikes(p: dict, has_r: bool) -> str:
    base = (
        "## ⚡ Interictal Spikes\n\n"
        "**Interictal spikes** are brief sharp EEG transients (<200 ms) occurring "
        "*between* seizures. They are a hallmark of epilepsy even during non-seizure periods.\n\n"
        "| Spike rate | Interpretation |\n"
        "|---|---|\n"
        "| < 0.1 / second | ✅ Normal |\n"
        "| 1–5 / second | ⚠️ Borderline — monitor closely |\n"
        "| > 5 / second | 🔴 Epileptiform — significant finding |\n"
        "| > 7 / second | 🔴 Focal ictal zone likely |\n\n"
        "**Patterns by seizure type:**\n"
        "- **Focal spikes** in one region → focal epilepsy\n"
        "- **3 Hz generalised spike-wave** → absence epilepsy\n"
        "- **High-amplitude polyspike** → tonic-atonic or myoclonic"
    )
    if has_r:
        si = p.get("spike_info") or {}
        n  = si.get("n_spikes", 0)
        r  = si.get("spike_rate_per_s", 0)
        if n > 0:
            level = "Normal" if r < 0.1 else ("Borderline ⚠️" if r < 5 else "Epileptiform 🔴")
            base += f"\n\n---\n📋 **Your EEG:** {n} spikes at {r:.2f}/s → **{level}**"
        else:
            base += "\n\n---\n📋 **Your EEG:** No spikes detected in this recording."
    return base


def _r_hjorth(ctx: str) -> str:
    return (
        "## 📊 Hjorth Parameters\n\n"
        "Hjorth parameters describe the **morphology** of the EEG signal using three descriptors:\n\n"
        "| Parameter | Measures | High value means |\n"
        "|---|---|---|\n"
        "| **Activity** | Signal variance (power) | High-energy signal |\n"
        "| **Mobility** | Mean frequency estimate | Fast oscillations |\n"
        "| **Complexity** | Deviation from a pure sine wave | Irregular waveform |\n\n"
        "**Typical values by class:**\n\n"
        "| Class | Activity | Mobility | Complexity |\n"
        "|---|---|---|---|\n"
        "| 🟢 Normal | Low | Medium | Low |\n"
        "| 🟠 Focal | Medium | High | Medium |\n"
        "| 🔴 Absence | High | Low | High (SWD peaks) |\n"
        "| 🟣 Tonic-Atonic | ⬆ Highest | ⬆ Highest | ⬆ Highest |\n"
        + ctx
    )


def _r_ai() -> str:
    return (
        "## 🤖 How NeuroScan Pro's AI Works\n\n"
        "**5-model soft-vote ensemble:**\n\n"
        "| Model | Role |\n"
        "|---|---|\n"
        "| **Random Forest** ⭐ | Primary — gold standard for EEG (Shoeb 2010, Ullah 2018) |\n"
        "| Logistic Regression | Linear baseline — fast and interpretable |\n"
        "| SVM (RBF kernel) | Non-linear boundary — strong EEG class separation |\n"
        "| KNN (k=21) | Distance baseline — intentionally weaker for comparison |\n"
        "| CNN-1D (NumPy) | Neural network — learns feature hierarchies without GPU |\n\n"
        "**Training protocol:**\n"
        "- **70%** train / **15%** validation / **15%** held-out test\n"
        "- **5-fold stratified cross-validation** for unbiased F1 estimates\n"
        "- Final prediction = **temperature-softened soft-vote** across all models with val F1 > 0.35\n"
        "- **RandomForest is always the primary model** (published clinical consensus)\n\n"
        "**Target accuracy:** 82–88% on 4-class EEG classification\n\n"
        "**Features used:** 46 clinical EEG features per recording — "
        "band power, entropy, Hjorth, wavelet, spikes, cross-correlation, fractal dimensions"
    )


def _r_confidence(p: dict, has_r: bool) -> str:
    base = (
        "## 🎯 AI Confidence Score\n\n"
        "The confidence score represents how strongly the **5-model ensemble** agrees on a prediction.\n\n"
        "| Confidence | Interpretation |\n"
        "|---|---|\n"
        "| **> 85%** | High confidence — result is very likely correct |\n"
        "| **70–85%** | Moderate — interpret with clinical context |\n"
        "| **55–70%** | Lower — treat as indicative, not definitive |\n"
        "| **< 55%** | Low — borderline case; further testing recommended |\n\n"
        "Confidence is computed after **temperature softening (T=1.5)** to prevent "
        "overconfident predictions — so 70% here is genuinely meaningful."
    )
    if has_r:
        conf  = p.get("confidence",0)*100
        level = "High ✅" if conf > 85 else ("Moderate ⚠️" if conf > 70 else "Lower — treat with caution ⚠️")
        base += f"\n\n---\n📋 **Your result confidence: {conf:.0f}% — {level}**"
    return base


def _r_epilepsy() -> str:
    return (
        "## 🧠 What is Epilepsy?\n\n"
        "**Epilepsy** is a chronic neurological disorder defined by "
        "**recurrent unprovoked seizures** caused by abnormal electrical discharges in the brain.\n\n"
        "**Key facts:**\n"
        "- Affects ~**50 million people** worldwide — 3rd most common neurological condition\n"
        "- Diagnosed after **2+ unprovoked seizures**, or 1 seizure with >60% recurrence risk\n"
        "- Classified as **focal** (one region) or **generalised** (both hemispheres)\n"
        "- ~**70% of patients** achieve seizure control with medication\n"
        "- **EEG is the gold standard** diagnostic tool\n\n"
        "**EEG** records the brain's electrical activity via scalp electrodes and reveals "
        "abnormal patterns (spike-wave discharges, focal theta bursts, rapid oscillations) "
        "that allow clinicians to classify the seizure type."
    )


def _r_eeg_def() -> str:
    return (
        "## 📡 What is an EEG?\n\n"
        "**EEG (Electroencephalography)** records the electrical activity of the brain "
        "using electrodes placed on the scalp.\n\n"
        "**How it works:**\n"
        "- Each electrode captures voltage changes from millions of neurons firing simultaneously\n"
        "- Standard clinical setup: **19 electrodes** (International 10–20 system)\n"
        "- Typical sampling rate: **256–512 Hz**\n"
        "- Signal filtered: band-pass 0.5–45 Hz + notch 50/60 Hz\n\n"
        "**What NeuroScan Pro extracts (46 features):**\n"
        "- Band powers (delta, theta, alpha, beta, gamma)\n"
        "- Entropy measures (sample, spectral, permutation)\n"
        "- Hjorth parameters (activity, mobility, complexity)\n"
        "- Wavelet features (DB4 decomposition, energy per sub-band)\n"
        "- Interictal spike rate and amplitude\n"
        "- Cross-channel correlation\n"
        "- Fractal dimensions (Higuchi, Katz), Lyapunov exponent, Hurst exponent"
    )


def _r_icd(ctx: str) -> str:
    return (
        "## 🏷️ ICD-10 Codes — Epilepsy\n\n"
        "| Classification | ICD-10 | Description |\n"
        "|---|---|---|\n"
        "| Normal EEG | — | No epileptic activity |\n"
        "| Focal seizure | **G40.1** | Localised onset, one hemisphere |\n"
        "| Absence seizure | **G40.3** | Childhood absence epilepsy (CAE) |\n"
        "| Tonic seizure | **G40.5** | Generalised tonic seizure |\n"
        "| Atonic seizure | **G40.8** | Atonic / drop attack seizure |\n\n"
        "ICD-10 codes are used for clinical documentation, insurance billing, and medical records. "
        "A neurologist assigns the final code after full clinical evaluation."
        + ctx
    )


def _r_causes() -> str:
    return (
        "## ⚠️ Causes & Triggers of Epilepsy\n\n"
        "**Underlying causes:**\n\n"
        "| Cause | Details |\n"
        "|---|---|\n"
        "| Genetic / hereditary | Idiopathic epilepsy — most common in absence and focal |\n"
        "| Structural | Tumour, stroke, traumatic brain injury, cortical malformation |\n"
        "| Metabolic | Hypoglycaemia, hyponatraemia, drug or alcohol withdrawal |\n"
        "| Infectious | Meningitis, encephalitis, neurocysticercosis |\n"
        "| Unknown | Cryptogenic — no identifiable cause found |\n\n"
        "**Seizure triggers** (in already-diagnosed epilepsy):\n"
        "- 😴 Sleep deprivation\n"
        "- 💊 Missed anti-seizure medication dose\n"
        "- 😰 Psychological stress or anxiety\n"
        "- 💡 Flickering lights (photosensitive epilepsy)\n"
        "- 🍺 Alcohol or recreational drug use\n"
        "- 🤒 Fever (especially febrile seizures in young children)\n"
        "- 🏃 Hyperventilation (strongly triggers absence seizures)"
    )


def _r_training() -> str:
    return (
        "## 🔬 NeuroScan Pro — Training Protocol\n\n"
        "**Data split:**\n\n"
        "| Set | Proportion | Purpose |\n"
        "|---|---|---|\n"
        "| Training | 70% | Fit all 5 models |\n"
        "| Validation | 15% | Model selection and threshold tuning |\n"
        "| Test (held-out) | 15% | Final unbiased accuracy benchmark |\n\n"
        "**Cross-validation:** 5-fold stratified CV for unbiased F1 estimates.\n\n"
        "**Data strategy:** Synthetic EEG features with clinically validated profiles "
        "are mixed with real data at a **4:1 synthetic-to-real ratio**. "
        "This overcomes the near-zero class separability in many real EEG feature CSVs.\n\n"
        "**No data leakage:** StandardScaler is fit on training set only, "
        "then applied identically to validation and test sets.\n\n"
        "**Target:** 82–88% weighted F1 — no overfitting (100%) or underfitting (<70%)."
    )


def _r_fallback(ctx: str) -> str:
    return (
        "🤔 **I'm not sure what you're asking.** Here are some things I can help with:\n\n"
        "| Category | Try asking... |\n"
        "|---|---|\n"
        "| 🔬 Your result | *\"What does my result mean?\"* |\n"
        "| ⚖️ Compare types | *\"Compare focal vs absence seizures\"* |\n"
        "| 🧠 Seizure info | *\"What is an absence seizure?\"* |\n"
        "| 💊 Medications | *\"What medications treat tonic-atonic?\"* |\n"
        "| 📋 Next steps | *\"What should I do after my diagnosis?\"* |\n"
        "| 🌊 EEG features | *\"What is entropy?\"*, *\"Explain band power\"* |\n"
        "| 🤖 AI system | *\"How does the AI work?\"*, *\"How accurate is it?\"* |\n\n"
        "Try rephrasing — I understand full sentences and natural language!"
        + ctx
    )
