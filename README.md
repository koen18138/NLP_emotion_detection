# Dutch Emotion Detection from YouTube Videos

End-to-end NLP pipeline that takes a Dutch YouTube URL and returns a per-sentence emotion classification (anger, disgust, fear, happiness, neutral, sadness, surprise). Built for a client (the "Content Intelligence Agency") who wanted automated emotion analysis of Dutch media without relying on paid third-party APIs.

This repo is the result of a multi-week NLP module. Each "Task" folder corresponds to a milestone (baselines → transformers → translation → prompt engineering → error analysis → XAI → model card). The final artefact is `pipeline.py`, which stitches the best components together.

---

## TL;DR

- **Input:** a Dutch-language YouTube URL.
- **Output:** a CSV (`data/pipeline_output.csv`) with sentence, timestamps, English translation, and predicted emotion.
- **Best model:** RoBERTa-base fine-tuned on GoEmotions (27 → 7 label remap) + extra minority-class samples.
- **Headline results on the 1042-sentence client test set:** 79.8% accuracy, 0.77 weighted F1, 0.54 macro F1 (limited by class imbalance — see [Results](#results-headline-numbers)).

---

## Pipeline overview

```
YouTube URL
   │
   ▼
[1] Audio download (pytubefix)           → data/audio/output.mp3
   │
   ▼
[2] ASR / transcription                  → DataFrame[Sentence, Start, End, (Speaker)]
       ├─ AssemblyAI (default: fast, accurate, gives timestamps + diarisation)
       └─ Whisper-large-v3 (offline fallback, slower, no per-sentence timestamps)
   │
   ▼
[3] Machine translation NL → EN          → "Translation" column
       Helsinki-NLP MarianMT (Helsinki-NLP/opus-mt-nl-en), batched, GPU if available
   │
   ▼
[4] Tokenise + clean text                → input_ids, attention_mask (max_len=128)
   │
   ▼
[5] Emotion classification               → "Emotion" column
       Fine-tuned RoBERTa-base (7-way softmax head)
   │
   ▼
[6] Save                                 → data/pipeline_output.csv
```

Entry point: [nlp_cia/src/pipeline.py](nlp_cia/src/pipeline.py). Each step is wrapped in `try/except` so a single failure (bad URL, missing API key, OOM) prints a clear error and stops the run.

### Why this shape

- **Translate to English before classification.** Public Dutch emotion datasets are tiny and skewed. Translating to English first lets us reuse the much larger English GoEmotions corpus and benefit from English-pretrained encoders. We tested the trade-off (translation artefacts vs. data scarcity) and translation won.
- **AssemblyAI as the default ASR.** Whisper-large-v3 is competitive on transcription quality, but the HuggingFace ASR pipeline doesn't give reliable sentence-level timestamps, which the downstream analysis needs. AssemblyAI gives sentence segmentation, timestamps and speaker labels in one call.
- **MarianMT for translation.** Free, runs locally, fast on a single GPU, and quality is good enough for emotion-bearing content. Avoids the per-call cost and rate limits of cloud translation.
- **RoBERTa over BERT for the classifier.** RoBERTa's training recipe consistently outperforms BERT on sentiment/affect tasks. We kept the pretrained encoder weights and re-initialised only the classification head (7 classes), so we benefit from transfer learning while specialising the output layer.

---

## What was actually done (and why)

### Task 6 — Modelling: baselines → transformers
We didn't start with a transformer. We built four classical baselines (Naive Bayes, Logistic Regression, SVM with TF-IDF features) and two neural baselines (RNN, LSTM) before fine-tuning a transformer. The point: have something to compare against so we can argue the transformer is actually worth the extra compute. The transformer (RoBERTa-base fine-tuned on GoEmotions) won decisively on macro F1, especially on minority classes.
- Notebooks: [`nlp_cia/task 6/`](nlp_cia/task%206/) (`naive.ipynb`, `Logistic.ipynb`, `SVM.ipynb`, `RNN.ipynb`, `LSTM.ipynb`, `pretrained_transformer_model.ipynb`).

### Task 7 — Machine translation
Trained a small NL→EN seq2seq from scratch as an exercise, then compared it against a pretrained MarianMT model. Pretrained won by a wide margin (as expected — we don't have parallel-corpus scale). MarianMT is what the production pipeline uses.
- Notebooks: [`nlp_cia/Task 7/`](nlp_cia/Task%207/) (`machine_translation.ipynb`, `machine_translation_pretrained.ipynb`).

### Task 8 — Prompt engineering as a comparison
Built a zero/few-shot LLM prompt for the same emotion task to sanity-check whether fine-tuning was actually worth it versus calling a general-purpose LLM. Fine-tuned model was cheaper at inference and competitive on the dominant classes; LLM prompting was better on rare classes but not cost-effective at scale.
- Files: [`nlp_cia/Task 8/`](nlp_cia/Task%208/).

### Task 9 — Error analysis
Built a confusion matrix and per-class breakdown of the fine-tuned RoBERTa. The model is good at majority classes (neutral, happiness) and unreliable for minorities (fear: recall 0.08, disgust: only 4 test samples). Root causes: class imbalance, translation artefacts ("een" → "one" changes meaning), and semantic overlap between subtle emotions.
- Report: [`nlp_cia/task 9/error_analysis_fine_tuned_pretrained_transformer_model.md`](nlp_cia/task%209/error_analysis_fine_tuned_pretrained_transformer_model.md).

### Task 10 — Explainable AI
Applied three XAI techniques (Gradient × Input, Conservative LRP from Ali et al. 2022, and input perturbation) to find out *what the model is actually looking at*. Key findings:
- The model does focus on emotional words ("fantastic", "shit", "terrible") — good.
- It's also disproportionately sensitive to the `<s>` aggregation token — an architectural quirk of RoBERTa classifiers, not a bug.
- Predictions are **fragile**: removing one key token from a short sentence can crash confidence from 55% to ~2%. The model leans on keywords more than we'd like.
- Full write-up: [`nlp_cia/Task 10/XAI_Analysis.md`](nlp_cia/Task%2010/XAI_Analysis.md).

### Task 11 — Model card
Full model card with intended use, out-of-scope use, dataset details, cultural caveats, sustainability notes (~4.5 kWh across all training runs on a single RTX 4070).
- File: [`nlp_cia/task 11/model card.md`](nlp_cia/task%2011/model%20card.md).

---

## Results (headline numbers)

Evaluated on the 1042-sentence held-out test set provided by the client:

| Metric | Value |
|---|---|
| Accuracy | 0.7975 |
| Weighted F1 | 0.7726 |
| Macro F1 | 0.5353 |

Per-class F1: neutral 0.88, happiness 0.72, anger 0.67, disgust 0.57, sadness 0.49, surprise 0.27, fear 0.15. The gap between weighted and macro F1 is the story — class imbalance dominates and the rare classes (fear, surprise) are unreliable.

---

## Known limitations (honest version)

- **Class imbalance.** Fear and disgust have very few training and test samples; their reported numbers are unreliable. The model over-predicts neutral.
- **Translation artefacts.** Some emotional nuance is lost when translating Dutch → English (Dutch directness, idioms, dialect). Reflected in error analysis.
- **Fragile to perturbations.** XAI shows confidence collapses on token removal for short sentences. Would likely struggle on typos and heavy paraphrasing.
- **Mild overfitting.** Validation loss starts rising after epoch 1 while training loss keeps dropping. Best checkpoint is likely epoch 1; we shipped epoch 3 for simplicity and would change that in production.
- **Slight scope.** 7 emotions is a coarse mapping from the original 27 GoEmotions labels — chosen with the client, but coarser than ideal for nuanced content.

What I'd do next: class-weighted loss / oversampling for minority classes, threshold tuning per class, audit the NL→EN translations on emotion-bearing test sentences, and pick the best-validation-F1 checkpoint instead of the final-epoch one.

---

## Repository layout

```
.
├── README.md                       ← you are here
├── Emotion pipeline Presentation.* ← slides used for the client presentation
├── Notebooks/                      ← exploratory notebooks
├── Transcriptions/                 ← example transcripts
├── data_iterations.md              ← notes from early data/model experiments
├── data_models/                    ← saved baseline artefacts
└── nlp_cia/                        ← the project itself
    ├── src/                        ← production pipeline code
    │   ├── pipeline.py             ← entry point
    │   ├── utils.py                ← YouTube → mp3
    │   ├── assembly.py             ← AssemblyAI ASR wrapper
    │   ├── whisper.py              ← Whisper ASR wrapper
    │   ├── machine_translation.py  ← MarianMT NL→EN
    │   ├── transformers_.py        ← model load / tokenise / predict
    │   ├── preprocessing.py        ← text cleaning helpers
    │   ├── feature_extraction.py   ← TF-IDF / classical feature extractors
    │   ├── RNN.py / multivariate_lstm.py  ← neural baselines
    │   └── sentiment_analysis_*.py ← earlier sentiment-task code
    ├── task 6/                     ← baselines + transformer fine-tuning
    ├── Task 7/                     ← machine translation experiments
    ├── Task 8/                     ← prompt-engineering comparison
    ├── task 9/                     ← error analysis
    ├── Task 10/                    ← XAI analysis
    ├── task 11/                    ← model card
    ├── models/                     ← saved models (download separately)
    ├── data/                       ← inputs and outputs
    └── pyproject.toml              ← Poetry dependencies
```

---

## How to run the pipeline

**1. Install dependencies (Poetry, Python 3.12).**
```
cd nlp_cia
poetry install
```

**2. Get the fine-tuned model.** Download from the OneDrive link shared in the project and place it at `nlp_cia/models/model_pretrained/`. Without this, step 5 of the pipeline will fail.

**3. Set an AssemblyAI API key** in `nlp_cia/src/assembly.py` (the `API_KEY` constant). Or pick Whisper at the prompt to run fully offline.

**4. Run.**
```
poetry run python src/pipeline.py
```

You'll be prompted for a YouTube URL and a transcription backend. Output goes to `nlp_cia/data/pipeline_output.csv`.

### Running the notebooks
```
cd nlp_cia
poetry install
poetry run python -m ipykernel install --user --name nlp-cia
```
Then select the `nlp-cia` kernel in your editor.

---

## Notes for reviewers

- The interesting code is in [`nlp_cia/src/`](nlp_cia/src/) (production pipeline) and the per-task notebooks under `nlp_cia/task*/`.
- The decision-making behind each component is in the model card ([`nlp_cia/task 11/model card.md`](nlp_cia/task%2011/model%20card.md)) and the XAI report ([`nlp_cia/Task 10/XAI_Analysis.md`](nlp_cia/Task%2010/XAI_Analysis.md)).
- The error analysis ([`nlp_cia/task 9/`](nlp_cia/task%209/)) is where I'd start if I were reviewing this work — it's the most honest summary of what the model does and doesn't do.
