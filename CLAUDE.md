# CLAUDE.md

Portfolio project for Madhavan T (data engineering / analytics job search). v2 rebuild of a 2022 university project (BSA Crescent Institute). Goal: an honest, well-evaluated ML project with a live browser demo, deployed on his Vercel portfolio and linked from LinkedIn.

## What's here
- `Toxic_Comment_Classifier_v2.ipynb`: the whole pipeline. Runs locally (VS Code) or in Colab (`IN_COLAB` switch in section 1–2).
- `scripts/download_data.py`: Kaggle API download into `data/` (needs `kaggle.json` + accepted competition rules).
- `web/index.html`: single-page project showcase with the live demo at the top. The demo fetches `web/toxic_model_web.json` and scores comments in plain JS.
- `toxic_clean.py`: `clean_text` + the `TextCleaner` pipeline step. Single source of truth, mirrored by the JS demo.
- `requirements-lock.txt`: `pip freeze` of the environment the reported numbers came from.
- `data/`, `models/`: git-ignored.

## Commands
- Setup: `python -m venv .venv`, activate, then `pip install -r requirements.txt`
- Data: `python scripts/download_data.py`
- Train + evaluate + export: run all cells of the notebook (writes `models/toxic_tfidf_logreg.joblib` and `web/toxic_model_web.json`)
- Preview site: `python -m http.server 8000 --directory web`
- Deploy: `vercel --cwd web` (deploy `web/` as the project root, not the repo root; `web/vercel.json` holds the config)

## Model
- Word TF-IDF (1–2 grams, `token_pattern=(?u)\b\w+\b`) + char_wb TF-IDF (2–5), both `sublinear_tf=True`, `strip_accents="unicode"`, l2 norm, 100K features each.
- `OneVsRestClassifier(LogisticRegression(C=4, solver="liblinear"))`. No `n_jobs` (liblinear + joblib memmaps raise "WRITEBACKIFCOPY base is read-only"). Section 7 fits four candidates and LinearSVC usually scores a hair higher on ROC-AUC, but LogReg is kept because thresholds, the demo and any moderation queue need `predict_proba`.
- Per-label thresholds maximise F1 on a 10% multilabel-stratified validation split.
- Final numbers come from the official Kaggle test set (`test.csv` + `test_labels.csv`, rows with -1 dropped).

## Rules that matter
- **Cleaning lives in `toxic_clean.py` only.** `clean_text` is defined there, imported by the notebook, and pickled into the saved pipeline as the `TextCleaner` step — so the artifact cleans its own input and `predict_proba` takes raw text. Never redefine it in the notebook.
- **The JS in `web/index.html` re-implements `clean_text` and sklearn's TfidfVectorizer.** If you change cleaning, tokenisation, n-gram ranges or vectorizer options, update both `toxic_clean.py` and the JS (`cleanText`, `stripAccents`, `wordGrams`, `charWbGrams`, `tfidf`) to match, then re-check parity: `window.__toxicScore(text)` in the browser console should match `pipeline.predict_proba([text])` to ~1e-4 (the gap comes from 16-bit weight quantisation). Last measured: 6.8e-05.
- Report results on the real class distribution only. Never on balanced resamples (that was the v1 mistake).
- Keep the HTML self-contained: inline CSS/JS, Google Fonts only.

## Results (2026-10-03 run, official Kaggle test set, 63,978 scored rows)
Mean column-wise ROC-AUC **0.9794**, mean PR-AUC 0.618, micro-F1 0.637, macro-F1 0.561. Per-label numbers are
in the README and in the `web/index.html` Results table. Fairness audit: 8.09% baseline false-positive rate,
but 5.8x that for non-toxic comments containing `gay`. Browser-export parity 6.8e-05.

## Open tasks
1. Replace the GitHub placeholder link in `web/index.html` (search `TODO`) once the repo exists; point the "Notebook" link at the v2 notebook.
2. Pushed to https://github.com/Madhavan12/toxic-comment-classifier (public). Vercel deploy is still pending: it needs an interactive `vercel login` first.
3. Update the LinkedIn project description with the real numbers (draft lives in the chat, not committed).
4. Colab needs `toxic_clean.py` beside the notebook, so the Colab path wants a `!git clone` line in section 1 once the repo URL exists.
5. Optional: run section 14 (DistilBERT) on a GPU and add the comparison.
6. Consider reducing identity-term false positives (augmentation, per-group thresholds, or the Jigsaw Unintended Bias dataset).
