"""Text cleaning for the toxic comment classifier.

This is the single source of truth for cleaning. Three things must agree:
  1. `clean_text` here, used by the notebook and by the saved pipeline
  2. `cleanText` in web/index.html, which re-implements it in JS for the demo
  3. the cleaning applied at training time

It lives in a module rather than in the notebook so that `TextCleaner` can be
pickled by reference and the saved joblib pipeline stays self-contained: the
model cleans its own input, and callers pass raw comment text.
"""
import re

from sklearn.base import BaseEstimator, TransformerMixin

URL_RE = re.compile(r"https?://\S+|www\.\S+")
IP_RE = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
USER_RE = re.compile(r"\[\[user:.*?\]\]", re.I)
WS_RE = re.compile(r"\s+")


def clean_text(s: str) -> str:
    """Lowercase, collapse URLs / IPs / user links to placeholders, squash whitespace.

    Idempotent: clean_text(clean_text(s)) == clean_text(s).
    """
    s = str(s).lower()
    s = URL_RE.sub(" url ", s)
    s = IP_RE.sub(" ip ", s)
    s = USER_RE.sub(" user ", s)
    s = s.replace("\n", " ")
    return WS_RE.sub(" ", s).strip()


class TextCleaner(BaseEstimator, TransformerMixin):
    """Applies `clean_text` to an iterable of comments.

    Stateless, so `fit` is a no-op. Exists as the first step of the saved
    pipeline, which is why raw text can be passed straight to `predict_proba`.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [clean_text(x) for x in X]
