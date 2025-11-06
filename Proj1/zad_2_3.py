import os
import string
import random
import time
from email import message_from_file
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np

# === KONFIGURACJA ===
INDEX_PATH = "trec07p/full/index"
DATA_PATH = "trec07p"
TRAIN_RATIO = 0.8
TOP_N = 100        # liczba słów w blacklist
SAMPLE_SIZE = None # ograniczenie liczby próbek, np. 2000 dla testów, None = całość
RESULTS_FILE = "results_stemming.txt"

# === FUNKCJE ===
def load_index(index_path):
    entries = []
    with open(index_path, "r") as f:
        for line in f:
            label, path = line.strip().split()
            full_path = os.path.join(DATA_PATH, path.replace("../", ""))
            entries.append((full_path, label))
    return entries

# Funcja do przetwarzania tekstu. Przeprowadza takie funkcje jak: czyszczenie, tokenizacja, usuwanie stopwords i (opcjonalnie) stemizacja
def preprocess_text(text, use_stemming=True):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if w not in stop_words and w.isalpha()]

    if use_stemming:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(w) for w in tokens]

    return tokens

# Wczytuje zawartość e-maila.
def load_email_content(filepath):
    try:
        with open(filepath, "r", encoding="latin-1") as f:
            msg = message_from_file(f)
            if msg.is_multipart():
                parts = [p.get_payload(decode=True) for p in msg.get_payload() if p.get_payload()]
                text = " ".join([str(p) for p in parts])
            else:
                text = msg.get_payload(decode=True)
            if text:
                text = text.decode(errors="ignore") if isinstance(text, bytes) else text
                return text
            else:
                return ""
    except Exception:
        return ""

# Tworzy listę słów kluczowych na podstawie danych treningowych.
def build_blacklist(train_data, top_n=100):
    spam_words = {}
    ham_words = {}
    for tokens, label in train_data:
        for token in tokens:
            if label == "spam":
                spam_words[token] = spam_words.get(token, 0) + 1
            else:
                ham_words[token] = ham_words.get(token, 0) + 1

    spam_ratio = {word: spam_words[word] / (ham_words.get(word, 0) + 1) for word in spam_words}
    sorted_words = sorted(spam_ratio.items(), key=lambda x: x[1], reverse=True)
    return [w for w, _ in sorted_words[:top_n]]

# Zwraca etykietę spam/ham w zależności od obecności słów zakazanych.
def classify_email(tokens, blacklist):
    return "spam" if any(word in blacklist for word in tokens) else "ham"

# Trenuje i testuje klasyfikator; zwraca accuracy, macierz konfuzji i czas.
def evaluate_model(train_entries, test_entries, use_stemming):
    start_time = time.time()

    train_data = []
    for path, label in train_entries:
        text = load_email_content(path)
        tokens = preprocess_text(text, use_stemming)
        train_data.append((tokens, label))

    blacklist = build_blacklist(train_data, top_n=TOP_N)

    y_true, y_pred = [], []
    for path, label in test_entries:
        text = load_email_content(path)
        tokens = preprocess_text(text, use_stemming)
        prediction = classify_email(tokens, blacklist)
        y_true.append(label)
        y_pred.append(prediction)

    elapsed = time.time() - start_time
    labels = ["spam", "ham"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_percent = cm / np.sum(cm) * 100
    acc = accuracy_score(y_true, y_pred) * 100

    return acc, cm_percent, elapsed


# === GŁÓWNY PROGRAM ===
def main():
    print("📂 Wczytywanie danych...")
    index_entries = load_index(INDEX_PATH)
    random.shuffle(index_entries)

    if SAMPLE_SIZE:
        index_entries = index_entries[:SAMPLE_SIZE]

    split_point = int(len(index_entries) * TRAIN_RATIO)
    train_entries = index_entries[:split_point]
    test_entries = index_entries[split_point:]

    results_log = []

    # Test 1: ZE STEMIZACJĄ
    print("🧠 Test 1: Z STEMIZACJĄ")
    acc_stem, cm_stem, time_stem = evaluate_model(train_entries, test_entries, use_stemming=True)
    print(f"🎯 Accuracy (stem): {acc_stem:.2f}% | ⏱ Czas: {time_stem:.2f}s")
    results_log.append(f"Test 1 (z stemizacją): accuracy={acc_stem:.2f}%, czas={time_stem:.2f}s")

    # Test 2: BEZ STEMIZACJI
    print("\n🧠 Test 2: BEZ STEMIZACJI")
    acc_no_stem, cm_no_stem, time_no_stem = evaluate_model(train_entries, test_entries, use_stemming=False)
    print(f"🎯 Accuracy (no stem): {acc_no_stem:.2f}% | ⏱ Czas: {time_no_stem:.2f}s")
    results_log.append(f"Test 2 (bez stemizacji): accuracy={acc_no_stem:.2f}%, czas={time_no_stem:.2f}s")

    # Porównanie wyników
    diff_acc = acc_stem - acc_no_stem
    diff_time = time_stem - time_no_stem

    summary = (
        "\n📊 PORÓWNANIE WYNIKÓW\n"
        f"Z stemizacją:    {acc_stem:.2f}% ({time_stem:.2f}s)\n"
        f"Bez stemizacji:  {acc_no_stem:.2f}% ({time_no_stem:.2f}s)\n"
        f"🧩 Różnica dokładności: {diff_acc:+.2f}%\n"
        f"⏱ Różnica czasu: {diff_time:+.2f}s (wartość dodatnia = wolniej ze stemizacją)\n"
    )

    print(summary)
    results_log.append(summary)

    # Macierze konfuzji
    matrix_report = (
        "\n📊 MACIERZ KONFUZJI (Z STEMIZACJĄ):\n"
        f"      spam      ham\n"
        f"spam  {cm_stem[0,0]:6.2f}%   {cm_stem[0,1]:6.2f}%\n"
        f"ham   {cm_stem[1,0]:6.2f}%   {cm_stem[1,1]:6.2f}%\n\n"
        "📊 MACIERZ KONFUZJI (BEZ STEMIZACJI):\n"
        f"      spam      ham\n"
        f"spam  {cm_no_stem[0,0]:6.2f}%   {cm_no_stem[0,1]:6.2f}%\n"
        f"ham   {cm_no_stem[1,0]:6.2f}%   {cm_no_stem[1,1]:6.2f}%\n"
    )
    print(matrix_report)
    results_log.append(matrix_report)

    # Zapis wyników do pliku
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(results_log))

    print(f"📁 Wyniki zapisano do pliku: {RESULTS_FILE}")


if __name__ == "__main__":
    main()