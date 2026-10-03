# Multilabel Toxic Comment Classification

Flags harmful comments across six non-exclusive categories (`toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`) using the [Jigsaw Toxic Comment Classification](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) dataset (~160K Wikipedia talk-page comments). Includes a web page where anyone can type a comment and score it with the trained model, running entirely in the browser.

## Approach
- **Features:** word (1–2 gram) + character (2–5 gram) TF-IDF, which handles misspellings and obfuscation like `f*ck`
- **Model:** one-vs-rest Logistic Regression, compared against Multinomial NB, word-only LogReg and LinearSVC.
  LinearSVC edges it on validation ROC-AUC (0.9860 vs 0.9854), inside run-to-run noise. LogReg is kept because
  the thresholds, the browser demo and any moderation queue all need `predict_proba`
- **Imbalance:** trained on the full imbalanced data, with multilabel iterative stratification for validation and a separate threshold tuned per label
- **Evaluation:** official Kaggle test set with its real class distribution, using mean column-wise ROC-AUC (the competition metric), PR-AUC and F1
- **Responsible AI:** top-n-gram explainability, error analysis, and an identity-term false-positive audit
- **Packaging:** a single sklearn `Pipeline` + thresholds saved with `joblib`, plus a browser export used by the live demo
- **Optional:** DistilBERT fine-tune for comparison (GPU)

## Results (official test set)
Mean column-wise ROC-AUC (the competition metric) is **0.9794** on the 63,978 scored test rows. Micro-F1 is 0.637 and macro-F1 0.561. Precision, recall and F1 use the per-label thresholds tuned for F1 on the validation split, so they trade precision for recall on the rare labels. ROC-AUC and PR-AUC are threshold-free. All of it is on the real class distribution, never a balanced resample.

Section 9b scores the identical predictions both ways to show why that matters: mean F1 is **0.820** on
balanced per-label samples against **0.561** on the real distribution. `severe_toxic` goes from 0.887 to
0.291. Only the test set changes between those numbers, and the v1 project reported the flattering one.

| Label | Positives in test | ROC-AUC | PR-AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| toxic | 6,090 | 0.965 | 0.784 | 0.531 | 0.870 | 0.659 |
| severe_toxic | 367 | 0.983 | 0.307 | 0.178 | 0.804 | 0.291 |
| obscene | 3,691 | 0.978 | 0.798 | 0.647 | 0.763 | 0.700 |
| threat | 211 | 0.993 | 0.514 | 0.482 | 0.583 | 0.528 |
| insult | 3,427 | 0.971 | 0.732 | 0.535 | 0.783 | 0.636 |
| identity_hate | 712 | 0.986 | 0.575 | 0.561 | 0.538 | 0.549 |
| **mean** | | **0.979** | **0.618** | **0.489** | **0.723** | **0.561** |

## Project layout
```
Toxic_Comment_Classifier_v2.ipynb   the full analysis and training
toxic_clean.py                      clean_text + TextCleaner, shared by notebook, model and JS demo
scripts/download_data.py            fetches the Kaggle data into data/
requirements-lock.txt               pinned versions behind the reported results
data/                               train.csv, test.csv, test_labels.csv (git-ignored)
models/                             joblib pipeline (git-ignored, regenerate from the notebook)
web/index.html                      project page + live demo
web/toxic_model_web.json            model export the demo loads (written by notebook section 13b)
web/vercel.json                     static-deploy config for serving web/ as the project root
```

## Run it locally (VS Code)
```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (source .venv/bin/activate on Mac/Linux)
pip install -r requirements.txt   # or requirements-lock.txt to pin the exact versions
python scripts/download_data.py   # needs kaggle.json and accepted competition rules
```
Open `Toxic_Comment_Classifier_v2.ipynb`, choose the `.venv` kernel, then **Run All**. The last full run took
**13.5 minutes** on a laptop CPU, most of it in section 6 (TF-IDF, 319s) and section 7 (fitting four candidate
models, 375s for the winner alone, because `liblinear` is single-threaded and the notebook sets no `n_jobs`).

`requirements.txt` holds minimum versions. `requirements-lock.txt` is a `pip freeze` of the environment that
produced the numbers above (Python 3.12.3, numpy 2.5.3, pandas 3.0.6, scikit-learn 1.9.1). Install from the
lock file to reproduce the table exactly.

Or run it headless:
```bash
jupyter nbconvert --to notebook --execute --inplace Toxic_Comment_Classifier_v2.ipynb
```

To try the demo page locally, serve the `web/` folder (opening the file directly can't load the model):
```bash
python -m http.server 8000 --directory web
```
then open http://localhost:8000.

## Deploy the demo
`web/` is a static site - the page plus the model JSON, no build step. Deploy that folder as the project root:

```bash
vercel login
vercel --cwd web            # preview
vercel --cwd web --prod     # production
```
Or import the repo at vercel.com and set **Root Directory** to `web`. Either way `web/vercel.json` applies, which
caches `toxic_model_web.json` for an hour with background revalidation - long enough that returning visitors
skip the 6 MB download, short enough that a retrained model reaches them.

## Use the model in Python
```python
import joblib
art = joblib.load("models/toxic_tfidf_logreg.joblib")   # needs toxic_clean.py importable
probs = art["pipeline"].predict_proba(["you are an idiot"])   # raw text: the pipeline cleans it
dict(zip(art["labels"], probs[0].round(3)))
# flag a label when it crosses its tuned threshold
[l for l, p in zip(art["labels"], probs[0]) if p >= art["thresholds"][l]]
```
The pipeline's first step is `TextCleaner` from `toxic_clean.py`, so pass raw comment text and let it clean.
That module has to be importable when you load the artifact, since the step is pickled by reference.

## Limitations
Measured on this run, not hypothetical:
- **Identity-term false positives.** Baseline false-positive rate on non-toxic comments is 8.09%, but it is
  47.2% for non-toxic comments containing `gay` (5.8x baseline), 22.7% for `woman` and 18.7% for `black`.
  This comes from the training data and is the strongest reason not to run this as an automated filter.
- **`severe_toxic` is weak at its operating point.** F1 0.291, precision 0.178: roughly five flags per true
  positive. ROC-AUC is 0.983, so the ranking is fine and only the threshold is bad - 367 positives in 63,978
  rows leaves little to tune on.
- **Bag-of-n-grams misses sarcasm and veiled threats** that contain no offensive words.
- **Thresholds are tuned for F1**, which favours recall. A real moderation queue would pick its own operating
  point per label, and probably a much higher precision one.
- **Wikipedia talk-page language** may not transfer to other platforms.
