#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integrated Fake News Detector (simplified, no synonyms.csv)
- Uses hardcoded synonyms for mapping
- Custom preprocessing: tokenization, synonym mapping, stemming, dynamic stopwords, unigrams+bigrams
- Custom TF/IDF and cosine similarity
- Custom Logistic Regression with L2
- Threshold tuning (ROC -> best F1)
- NewsAPI verification (TF-IDF cosine + semantic SBERT similarity)
- CLI: --auto_download, --train, --detect
"""

import os
import re
import csv
import json
import math
import time
import argparse

import logging
import random
import requests
import numpy as np
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

# third-party libs (make sure installed)
import kaggle
from cachetools import TTLCache
from nltk.stem.porter import PorterStemmer
from sentence_transformers import SentenceTransformer
from sklearn.metrics import roc_curve
import pandas as pd

# ---------------------------
# Logging
# ---------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ---------------------------
# Globals
# ---------------------------
NEWS_CACHE = TTLCache(maxsize=200, ttl=3600)
ps = PorterStemmer()
try:
    EMBEDDER = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    logging.warning(f"SentenceTransformer load failed ({e}). Semantic similarity disabled.")
    EMBEDDER = None

DYNAMIC_STOPWORDS = set()

# Hardcoded synonyms
HARDCODED_SYNONYMS = {
    'angry': ['enraged', 'indignant', 'outraged', 'infuriated', 'irate', 'furious', 'mad', 'livid'],
    'happy': ['joyful', 'delighted', 'pleased', 'glad', 'cheerful', 'ecstatic', 'thrilled', 'elated'],
    'sad': ['sorrowful', 'mournful', 'depressed', 'melancholy', 'gloomy', 'heartbroken', 'downcast'],
    'fear': ['scared', 'afraid', 'terrified', 'frightened', 'petrified', 'panicked', 'anxious'],
    'good': ['great', 'excellent', 'wonderful', 'fantastic', 'superb', 'amazing', 'terrific'],
    'bad': ['terrible', 'awful', 'horrible', 'dreadful', 'poor', 'lousy', 'atrocious'],
    'big': ['large', 'huge', 'enormous', 'gigantic', 'massive', 'immense', 'vast'],
    'small': ['tiny', 'little', 'miniature', 'petite', 'minute', 'microscopic'],
    'say': ['tell', 'speak', 'state', 'declare', 'announce', 'express', 'utter'],
    'think': ['believe', 'consider', 'ponder', 'reflect', 'contemplate', 'suppose']
}

# ---------------------------
# Synonym lookup builder
# ---------------------------
def build_synonym_lookup():
    """
    Build a reverse lookup map: any word -> canonical lemma using hardcoded synonyms.
    """
    lookup = {}
    for lemma, syns in HARDCODED_SYNONYMS.items():
        lemma_norm = lemma.lower()
        lookup[lemma_norm] = lemma_norm
        for s in syns:
            lookup[s.lower()] = lemma_norm
    return lookup

SYNONYM_LOOKUP = build_synonym_lookup()

def map_to_synonym(word):
    """
    Map a word to its canonical lemma if present, otherwise return the original word.
    Lowercases the input.
    """
    if not word:
        return word
    w = word.lower()
    return SYNONYM_LOOKUP.get(w, w)

# ---------------------------
# Text utilities & preprocessing
# ---------------------------
def custom_html_to_text(html: str) -> str:
    text = re.sub(r'<[^>]*>', ' ', html)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def custom_tokenize(text: str, n_grams: int = 2):
    text = (text or "").lower()
    punctuation = ".,!?;:'\"()[]{}–—*/\\&%$#@+=<>~`|^_—–"
    for ch in punctuation:
        text = text.replace(ch, ' ')
    tokens = [t for t in text.split() if t]
    ngrams = []
    for i in range(len(tokens)):
        ngrams.append(tokens[i])  # unigram
        if i < len(tokens) - 1 and n_grams >= 2:
            ngrams.append(f"{tokens[i]}_{tokens[i+1]}")  # bigram
    return ngrams

def generate_dynamic_stopwords(documents, freq_threshold=0.5):
    doc_count = len(documents)
    if doc_count == 0:
        return set()
    word_doc_count = Counter()
    for doc in documents:
        tokens = set(custom_tokenize(doc, n_grams=1))
        word_doc_count.update(tokens)
    return {w for w, c in word_doc_count.items() if (c / doc_count) >= freq_threshold}

STATIC_STOPWORDS = {
    'a','about','above','across','after','again','against','all','almost','alone','along','already','also','although','always',
    'am','among','an','and','another','any','are','as','at','be','because','been','before','being','below','between','both','but',
    'by','can','could','did','do','does','doing','down','during','each','few','for','from','further','had','has','have','having',
    'he','her','here','hers','herself','him','himself','his','how','i','if','in','into','is','it','its','itself','just','me',
    'more','most','my','myself','no','nor','not','of','off','on','once','only','or','other','our','ours','ourselves','out','over',
    'own','same','she','should','so','some','such','than','that','the','their','theirs','them','themselves','then','there','these',
    'they','this','those','through','to','too','under','until','up','very','was','we','were','what','when','where','which','while',
    'who','whom','why','with','would','you','your','yours','yourself','yourselves'
}

def custom_remove_stopwords(tokens):
    stopwords = STATIC_STOPWORDS | DYNAMIC_STOPWORDS
    return [t for t in tokens if t not in stopwords]

def custom_stem(word: str) -> str:
    return ps.stem(word) if len(word) >= 3 else word

def custom_preprocess_tokens(tokens):
    tokens = custom_remove_stopwords(tokens)
    tokens = [custom_stem(t) for t in tokens]
    tokens = [map_to_synonym(t) for t in tokens]
    return [t for t in tokens if t]

# ---------------------------
# TF / IDF / TF-IDF
# ---------------------------
def custom_tf(text: str, n_grams=2):
    if not text or not text.strip():
        raise ValueError("Empty text for TF.")
    tokens = custom_preprocess_tokens(custom_tokenize(text, n_grams))
    if not tokens:
        raise ValueError("No valid tokens after preprocessing.")
    cnt = Counter(tokens)
    total = len(tokens)
    return {w: c / total for w, c in cnt.items()}

def custom_idf(documents, n_grams=2):
    doc_count = len(documents)
    word_doc_count = {}
    for doc in documents:
        toks = set(custom_preprocess_tokens(custom_tokenize(doc, n_grams)))
        for t in toks:
            word_doc_count[t] = word_doc_count.get(t, 0) + 1
    return {w: math.log((doc_count) / (c + 1)) for w, c in word_doc_count.items()}

def custom_tfidf(documents, n_grams=2):
    idf = custom_idf(documents, n_grams)
    out = []
    for doc in documents:
        tf = custom_tf(doc, n_grams)
        out.append({w: tfv * idf.get(w, 0.0) for w, tfv in tf.items()})
    return out

def custom_cosine_similarity(vec1: dict, vec2: dict) -> float:
    inter = set(vec1.keys()) & set(vec2.keys())
    if not inter:
        return 0.0
    dot = sum(vec1[k] * vec2[k] for k in inter)
    n1 = math.sqrt(sum(v*v for v in vec1.values()))
    n2 = math.sqrt(sum(v*v for v in vec2.values()))
    if n1 == 0 or n2 == 0:
        return 0.0
    return dot / (n1 * n2)

# ---------------------------
# Custom Logistic Regression
# ---------------------------
class CustomLogisticRegression:
    def __init__(self, learning_rate=0.1, max_iterations=1000, l2_lambda=0.01):
        self.weights = {}
        self.bias = 0.0
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.l2_lambda = l2_lambda

    @staticmethod
    def sigmoid(z):
        z = max(min(z, 500), -500)
        return 1.0 / (1.0 + math.exp(-z))

    def fit(self, X_list, y):
        vocab = set()
        for x in X_list:
            vocab.update(x.keys())
        self.weights = {w: 0.0 for w in vocab}
        self.bias = 0.0

        n = len(X_list)
        for _ in range(self.max_iterations):
            grad_w = {w: 0.0 for w in self.weights}
            grad_b = 0.0
            for x, label in zip(X_list, y):
                z = sum(self.weights.get(w, 0.0) * val for w, val in x.items()) + self.bias
                y_hat = self.sigmoid(z)
                err = y_hat - label
                for w, val in x.items():
                    grad_w[w] += err * val + self.l2_lambda * self.weights[w]
                grad_b += err
            for w in self.weights:
                self.weights[w] -= self.learning_rate * (grad_w[w] / n)
            self.bias -= self.learning_rate * (grad_b / n)

    def predict_proba(self, X_list):
        probs = []
        for x in X_list:
            z = sum(self.weights.get(w, 0.0) * val for w, val in x.items()) + self.bias
            probs.append(self.sigmoid(z))
        return probs

    def predict(self, X_list, threshold=0.5):
        return [1 if p >= threshold else 0 for p in self.predict_proba(X_list)]

    def save_model(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'bias': self.bias, 'weights': self.weights}, f, indent=2)

    @classmethod
    def load_model(cls, path, learning_rate=0.1, max_iterations=1000, l2_lambda=0.01):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            m = cls(learning_rate, max_iterations, l2_lambda)
            m.bias = data['bias']
            m.weights = data['weights']
            return m
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            return None

# ---------------------------
# Metrics + train/test split
# ---------------------------
def custom_accuracy(y_true, y_pred):
    return sum(int(a == b) for a, b in zip(y_true, y_pred)) / len(y_true) if y_true else 0.0

def custom_precision(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    pp = sum(1 for p in y_pred if p == 1)
    return tp / pp if pp else 0.0

def custom_recall(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    pos = sum(1 for t in y_true if t == 1)
    return tp / pos if pos else 0.0

def custom_f1(y_true, y_pred):
    p = custom_precision(y_true, y_pred)
    r = custom_recall(y_true, y_pred)
    return 2*p*r/(p+r) if (p+r) else 0.0

def custom_confusion_matrix(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    return [[tn, fp], [fn, tp]]

def custom_train_test_split(data, labels, test_size=0.2, seed=42):
    random.seed(seed)
    pairs = list(zip(data, labels))
    random.shuffle(pairs)
    cut = int(len(pairs) * (1 - test_size))
    train = pairs[:cut]
    test = pairs[cut:]
    X_train, y_train = [x for x, _ in train], [y for _, y in train]
    X_test, y_test = [x for x, _ in test], [y for _, y in test]
    return X_train, X_test, y_train, y_test

def tune_thresholds(model, X_val, y_val):
    probs = model.predict_proba(X_val)
    try:
        fpr, tpr, thresholds = roc_curve(y_val, probs)
    except Exception:
        return 0.5
    best_f1, best_t = 0.0, 0.5
    for t in thresholds:
        preds = [1 if p >= t else 0 for p in probs]
        f1 = custom_f1(y_val, preds)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    logging.info(f"Validation: best F1={best_f1:.4f} at threshold={best_t:.4f}")
    return best_t

# ---------------------------
# CSV loader for datasets
# ---------------------------
def load_news_rows(file_path, assume_label=None):
    rows = []
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f)
        header = next(reader)
        header_lower = [h.strip().lower() for h in header]
        def idx(name):
            return header_lower.index(name) if name in header_lower else None
        ti, xi, li = idx('title'), idx('text'), idx('label')
        if ti is None or xi is None:
            raise ValueError(f"{file_path} must have 'title' and 'text' columns.")
        for row in reader:
            if len(row) <= max(ti, xi):
                continue
            title = row[ti].strip()
            text = row[xi].strip()
            if li is not None and len(row) > li:
                try:
                    label = int(row[li])
                except:
                    continue
            else:
                if assume_label is None:
                    raise ValueError(f"No label column and no assume_label provided for {file_path}")
                label = int(assume_label)
            rows.append({'title': title, 'text': text, 'label': label})
    return rows

# ---------------------------
# Train pipeline
# ---------------------------
def train_model(fake_csv_path, true_csv_path, model_path='fake_news_model.json'):
    fake_rows = load_news_rows(fake_csv_path, assume_label=0)
    true_rows = load_news_rows(true_csv_path, assume_label=1)
    all_rows = fake_rows + true_rows

    texts = [r['title'] + ' ' + r['text'] for r in all_rows]
    labels = [r['label'] for r in all_rows]

    global DYNAMIC_STOPWORDS
    DYNAMIC_STOPWORDS = generate_dynamic_stopwords(texts, freq_threshold=0.5)
    logging.info(f"Dynamic stopwords size: {len(DYNAMIC_STOPWORDS)}")

    try:
        tfidf_docs = custom_tfidf(texts, n_grams=2)
    except Exception as e:
        logging.error(f"TF-IDF error: {e}")
        return None

    X_train_val, X_test, y_train_val, y_test = custom_train_test_split(tfidf_docs, labels, test_size=0.2, seed=42)
    X_train, X_val, y_train, y_val = custom_train_test_split(X_train_val, y_train_val, test_size=0.2, seed=42)

    model = CustomLogisticRegression(learning_rate=0.1, max_iterations=1000, l2_lambda=0.01)
    model.fit(X_train, y_train)

    optimal_t = tune_thresholds(model, X_val, y_val)

    y_pred = model.predict(X_test, threshold=optimal_t)
    acc = custom_accuracy(y_test, y_pred)
    prec = custom_precision(y_test, y_pred)
    rec = custom_recall(y_test, y_pred)
    f1 = custom_f1(y_test, y_pred)
    cm = custom_confusion_matrix(y_test, y_pred)

    print("=== Test Evaluation ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"Confusion Matrix: [[TN={cm[0][0]}, FP={cm[0][1]}],[FN={cm[1][0]}, TP={cm[1][1]}]]")
    print(f"Optimal Threshold: {optimal_t:.4f}")

    model.save_model(model_path)
    meta = {'optimal_threshold': optimal_t, 'dynamic_stopwords_size': len(DYNAMIC_STOPWORDS)}
    with open(model_path.replace('.json', '_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    logging.info(f"Model saved to {model_path}")
    return model, optimal_t

# ---------------------------
# NewsAPI verification
# ---------------------------
def fetch_news(api_key, query, days_back=7):
    if not api_key or not query:
        logging.error("Invalid api_key or query.")
        return []
    cache_key = f"{query}_{days_back}"
    if cache_key in NEWS_CACHE:
        return NEWS_CACHE[cache_key]
    url = "https://newsapi.org/v2/everything"
    from_date = (datetime.now() - timedelta(days=days_back)).date().isoformat()
    domains = 'timesofindia.indiatimes.com,bbc.com,cnn.com,nytimes.com,reuters.com,apnews.com,theguardian.com'
    params = {'q': query[:128], 'domains': domains, 'from': from_date, 'language': 'en',
              'sortBy': 'relevancy', 'pageSize': 20, 'apiKey': api_key}
    try:
        r = requests.get(url, params=params, timeout=12)
        r.raise_for_status()
        data = r.json()
        if data.get('status') != 'ok':
            logging.error(f"NewsAPI error: {data.get('message')}")
            return []
        articles = data.get('articles', [])
        NEWS_CACHE[cache_key] = articles
        return articles
    except Exception as e:
        logging.error(f"News fetch error: {e}")
        return []

def semantic_similarity(a: str, b: str) -> float:
    if EMBEDDER is None:
        return 0.0
    emb = EMBEDDER.encode([a, b], convert_to_numpy=True)
    v1, v2 = emb[0], emb[1]
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2))
    return float(np.dot(v1, v2) / denom) if denom else 0.0

def fetch_article_text(article):
    title = (article.get('title') or '').strip()
    desc = (article.get('description') or '').strip()
    content = (article.get('content') or '').strip()
    url = article.get('url')
    text = f"{title} {desc} {content}".strip()
    return text, title, url

def detect_fake_news(user_news, api_key, model_path='fake_news_model.json', sim_threshold=0.30, ml_threshold=None):
    logging.info("Detection started.")
    if not user_news or not user_news.strip():
        return "Error: empty news text."
    if len(user_news.split()) < 10:
        return "Error: news text too short (min 10 words)."

    if not api_key:
        logging.warning("No API key provided; NewsAPI checks will fail.")

    model = CustomLogisticRegression.load_model(model_path)
    if model is None:
        return "Error: model not found. Train first."

    meta_path = model_path.replace('.json', '_meta.json')
    th_opt = None
    if os.path.exists(meta_path):
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            th_opt = meta.get('optimal_threshold')
        except Exception:
            pass
    if ml_threshold is None:
        ml_threshold = th_opt if th_opt is not None else 0.5

    try:
        user_tfidf = custom_tf(user_news, n_grams=2)
    except Exception as e:
        return f"Error processing input: {e}"

    prob = model.predict_proba([user_tfidf])[0]
    ml_pred = 1 if prob >= ml_threshold else 0

    toks = custom_preprocess_tokens(custom_tokenize(user_news, n_grams=1))
    uniq = list(dict.fromkeys(toks))
    search_query = ' '.join(uniq[:20]) or user_news[:100]

    articles = fetch_news(api_key, search_query) if api_key else []
    if not articles:
        return (f"No corroborating articles found.\nML says: {'REAL' if ml_pred==1 else 'FAKE'} (p={prob:.2f}).")

    best = {'tfidf_sim': 0.0, 'semantic_sim': 0.0, 'title': None, 'url': None}

    for text, title, url in ThreadPoolExecutor(max_workers=6).map(fetch_article_text, articles):
        try:
            art_tfidf = custom_tf(text, n_grams=2)
        except Exception:
            continue
        tfidf_sim = custom_cosine_similarity(user_tfidf, art_tfidf)
        sem_sim = semantic_similarity(user_news, text) if EMBEDDER is not None else 0.0
        if (tfidf_sim + sem_sim) > (best['tfidf_sim'] + best['semantic_sim']):
            best.update({'tfidf_sim': tfidf_sim, 'semantic_sim': sem_sim, 'title': title, 'url': url})

    looks_real = (best['tfidf_sim'] >= sim_threshold) or (best['semantic_sim'] >= sim_threshold) or (prob >= ml_threshold)
    verdict = "REAL ✅" if looks_real else "FAKE ❌"
    details = [
        f"Verdict: {verdict}",
        f"ML Probability: {prob:.2f} (threshold={ml_threshold:.2f})",
        f"TF-IDF Similarity: {best['tfidf_sim']:.2f}",
        f"Semantic Similarity: {best['semantic_sim']:.2f}"
    ]
    if best['title'] and best['url']:
        details.append(f"Best match: {best['title']} ({best['url']})")
    return "\n".join(details)

# ---------------------------
# Kaggle dataset auto-download helper
# ---------------------------
def download_kaggle_dataset_if_needed(dataset_slug='clmentbisaillon/fake-and-real-news-dataset', zip_name='fake-and-real-news-dataset.zip', expect_files=('Fake.csv', 'True.csv')):
    have_all = all(os.path.exists(f) for f in expect_files)
    if have_all:
        logging.info("Dataset already present; skipping download.")
        return
    if not os.path.exists(zip_name):
        logging.info("Downloading Kaggle dataset...")
        code = os.system(f'kaggle datasets download -d {dataset_slug} -p .')
        if code != 0:
            raise RuntimeError("Kaggle download failed. Ensure kaggle is installed and ~/.kaggle/kaggle.json configured.")
    logging.info("Extracting dataset...")
    with zipfile.ZipFile(zip_name, 'r') as zf:
        zf.extractall('.')
    logging.info("Extraction complete.")

# ---------------------------
# CLI
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Integrated Fake News Detector")
    parser.add_argument('--auto_download', action='store_true', help="Auto-download Kaggle dataset")
    parser.add_argument('--train', action='store_true', help="Train model")
    parser.add_argument('--fake_csv', type=str, default='Fake.csv', help="Path to Fake.csv")
    parser.add_argument('--true_csv', type=str, default='True.csv', help="Path to True.csv")
    parser.add_argument('--model_path', type=str, default='fake_news_model.json', help="Model output file")
    parser.add_argument('--detect', type=str, help="News text to detect")
    parser.add_argument('--api_key', type=str, help="NewsAPI key for verification")
    parser.add_argument('--sim_threshold', type=float, default=0.30, help="Similarity threshold")
    parser.add_argument('--ml_threshold', type=float, default=None, help="Optional ML threshold override")
    args = parser.parse_args()

    if args.auto_download:
        download_kaggle_dataset_if_needed()

    if args.train:
        try:
            model_info = train_model(args.fake_csv, args.true_csv, args.model_path)
            if model_info is None:
                raise SystemExit(1)
        except Exception as e:
            logging.error(f"Training failed: {e}")
            raise

    if args.detect:
        if not args.api_key:
            logging.warning("No NewsAPI key provided; detection will skip NewsAPI checks.")
        result = detect_fake_news(user_news=args.detect, api_key=args.api_key, model_path=args.model_path, sim_threshold=args.sim_threshold, ml_threshold=args.ml_threshold)
        print(result)

    if not args.train and not args.detect:
        parser.print_help()

if __name__ == "__main__":
    main()