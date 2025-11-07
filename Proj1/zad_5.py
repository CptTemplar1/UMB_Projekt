import os
import string
import random
import time
from email import message_from_file

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np

# === KONFIGURACJA ===
INDEX_PATH = "trec07p/full/index"       # ścieżka do indexu
DATA_PATH = "trec07p"                   # ścieżka do danych
TRAIN_RATIO = 0.8                       # stosunek danych treningowych do testowych
SAMPLE_SIZE = None                      # ograniczenie liczby próbek, np. 2000 dla testów, None = całość
RESULTS_FILE = "results_naive_bayes.txt"# nazwa pliku wynikowego

random.seed(42)                         # ustawienie ziarna losowości


# === POMOCNICZE FUNKCJE ===
def load_index(index_path):
    entries = []
    with open(index_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                label, path = parts[0], parts[1]
                full_path = os.path.join(DATA_PATH, path.replace("../", ""))
                entries.append((full_path, label))
    return entries

# Wczytuje treść e-maila (temat + ciało) jako zwykły tekst
def load_email_content(filepath):
    try:
        with open(filepath, "r", encoding="latin-1") as f:
            msg = message_from_file(f)
            subject = msg.get("Subject", "")
            payload = ""
            if msg.is_multipart():
                parts = []
                for part in msg.walk():
                    ctype = part.get_content_type()
                    if ctype.startswith("text/"):
                        p = part.get_payload(decode=True)
                        if p:
                            parts.append(p)
                payload = " ".join(str(p) for p in parts)
            else:
                p = msg.get_payload(decode=True)
                payload = p if p else ""
            if isinstance(payload, bytes):
                payload = payload.decode(errors="ignore")
            return subject + " " + payload
    except Exception:
        return ""

# Usuwa interpunkcję, stopwords i dokonuje stemizacji
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    sw = set(stopwords.words("english"))
    tokens = [t for t in tokens if t.isalpha() and t not in sw]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(t) for t in tokens]
    return " ".join(tokens)

# Zwraca listę tekstów i etykiet (spam/ham), z opcjonalnym preprocessingiem.
def prepare_data(entries, use_preprocessing=False):
    texts, labels = [], []
    for path, label in entries:
        text = load_email_content(path)
        if use_preprocessing:
            text = preprocess_text(text)
        texts.append(text)
        labels.append(label)
    return texts, labels


# === FUNKCJA EKSPERYMENTU ===
#  Trenuje i testuje klasyfikator MultinomialNB dla zbioru TREC07P. Zwraca accuracy, macierz konfuzji i czas wykonania.
def run_naive_bayes(train_entries, test_entries, use_preprocessing=False):
    print(f"\n🧠 Uruchamianie Naive Bayes ({'z preprocessingiem' if use_preprocessing else 'Bez preprocessingu'})...")
    start_time = time.time()

    # Przygotowanie danych
    X_train_texts, y_train = prepare_data(train_entries, use_preprocessing)
    X_test_texts, y_test = prepare_data(test_entries, use_preprocessing)

    # Konwersja do macierzy cech (bag of words)
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(X_train_texts)
    X_test = vectorizer.transform(X_test_texts)

    # Trening
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # Predykcja
    y_pred = model.predict(X_test)

    # Ewaluacja
    elapsed = time.time() - start_time
    labels = ["spam", "ham"]
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_percent = cm / np.sum(cm) * 100
    acc = accuracy_score(y_test, y_pred) * 100

    # Wyświetlenie wyników
    print(f"🎯 Accuracy: {acc:.2f}% | ⏱ Czas wykonania: {elapsed:.2f}s")
    print("📊 Confusion matrix (%):")
    print(f"      spam      ham")
    print(f"spam  {cm_percent[0,0]:6.2f}%   {cm_percent[0,1]:6.2f}%")
    print(f"ham   {cm_percent[1,0]:6.2f}%   {cm_percent[1,1]:6.2f}%")

    return acc, cm_percent, elapsed


# === GŁÓWNY PROGRAM ===
def main():
    print("📂 Wczytywanie danych...")
    index_entries = load_index(INDEX_PATH)
    random.shuffle(index_entries)

    if SAMPLE_SIZE:
        index_entries = index_entries[:SAMPLE_SIZE]
        print(f"⚠️ SAMPLE_SIZE aktywne. Wykorzystuję {len(index_entries)} pierwszych wpisów")

    split_point = int(len(index_entries) * TRAIN_RATIO)
    train_entries = index_entries[:split_point]
    test_entries = index_entries[split_point:]
    print(f"Łącznie: {len(index_entries)} dokumentów; trening: {len(train_entries)}; test: {len(test_entries)}")

    # Wyniki
    results = []

    # Wersja bez preprocessingu (pełny tekst)
    acc_raw, cm_raw, t_raw = run_naive_bayes(train_entries, test_entries, use_preprocessing=False)
    results.append(("Bez preprocessingu", acc_raw, cm_raw, t_raw))

    # Wersja z preprocessingiem (usuwanie stopwords i stemizacja)
    acc_clean, cm_clean, t_clean = run_naive_bayes(train_entries, test_entries, use_preprocessing=True)
    results.append(("Z preprocessingiem (NLTK)", acc_clean, cm_clean, t_clean))

    # Zapis wyników do pliku
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        f.write("Naive Bayes Results\n\n")
        for title, acc, cm, t in results:
            f.write(f"{title}\n")
            f.write(f"Accuracy: {acc:.2f}%\nCzas: {t:.2f}s\n")
            f.write("Confusion matrix (%):\n")
            f.write(f"spam_spam={cm[0,0]:.2f}% spam_ham={cm[0,1]:.2f}%\n")
            f.write(f"ham_spam={cm[1,0]:.2f}% ham_ham={cm[1,1]:.2f}%\n\n")

    print(f"\n📁 Wyniki zapisano do: {RESULTS_FILE}")

if __name__ == "__main__":
    main()