import os
import string
import random

#Potrzebne tylko przy pierwszym uruchomieniu
#import nltk
#nltk.download('stopwords')
#nltk.download('punkt_tab')

from email import message_from_file
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np

# === KONFIGURACJA ===
INDEX_PATH = "trec07p/full/index"
DATA_PATH = "trec07p"
TRAIN_RATIO = 0.8  # 80% trening, 20% test

# === FUNKCJE POMOCNICZE ===
def load_index(index_path):
    """Wczytuje plik index i zwraca listę (ścieżka, etykieta)."""
    entries = []
    with open(index_path, "r") as f:
        for line in f:
            label, path = line.strip().split()
            # Normalizuj ścieżkę
            full_path = os.path.join(DATA_PATH, path.replace("../", ""))
            entries.append((full_path, label))
    return entries


def preprocess_text(text):
    """Czyszczenie, tokenizacja, usuwanie stopwords i stemizacja."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    cleaned_tokens = [stemmer.stem(w) for w in tokens if w not in stop_words and w.isalpha()]
    return cleaned_tokens


def load_email_content(filepath):
    """Wczytuje zawartość pliku e-mail."""
    try:
        with open(filepath, "r", encoding="latin-1") as f:
            msg = message_from_file(f)
            if msg.is_multipart():
                parts = [part.get_payload(decode=True) for part in msg.get_payload() if part.get_payload()]
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


def build_blacklist(train_data, top_n=100):
    """Tworzy listę słów kluczowych (blacklist) na podstawie zbioru treningowego."""
    spam_words = {}
    ham_words = {}
    for text, label in train_data:
        for token in text:
            if label == "spam":
                spam_words[token] = spam_words.get(token, 0) + 1
            else:
                ham_words[token] = ham_words.get(token, 0) + 1

    # Oblicz wskaźnik spamowości słowa
    spam_ratio = {}
    for word in spam_words:
        spam_count = spam_words.get(word, 0)
        ham_count = ham_words.get(word, 0)
        ratio = spam_count / (ham_count + 1)
        spam_ratio[word] = ratio

    # Sortuj według największego udziału w spamie
    sorted_words = sorted(spam_ratio.items(), key=lambda x: x[1], reverse=True)
    blacklist = [w for w, r in sorted_words[:top_n]]
    return blacklist


def classify_email(tokens, blacklist):
    """Klasyfikacja binarna na podstawie obecności słów z blacklisty."""
    return "spam" if any(word in blacklist for word in tokens) else "ham"


# === GŁÓWNY PROGRAM ===
def main():
    print("📂 Wczytywanie danych...")
    index_entries = load_index(INDEX_PATH)[:2000]  # Ograniczenie do 2000 dla szybkości
    random.shuffle(index_entries)

    split_point = int(len(index_entries) * TRAIN_RATIO)
    train_entries = index_entries[:split_point]
    test_entries = index_entries[split_point:]

    train_data = []
    test_data = []

    print("🧠 Przetwarzanie zbioru treningowego...")
    for path, label in train_entries:
        text = load_email_content(path)
        tokens = preprocess_text(text)
        train_data.append((tokens, label))

    print("🧠 Generowanie blacklisty...")
    blacklist = build_blacklist(train_data, top_n=100)
    print(f"📜 Lista słów zakazanych ({len(blacklist)}): {blacklist[:15]} ...")

    print("🧪 Ewaluacja na zbiorze testowym...")
    y_true = []
    y_pred = []

    for path, label in test_entries:
        text = load_email_content(path)
        tokens = preprocess_text(text)
        prediction = classify_email(tokens, blacklist)
        y_true.append(label)
        y_pred.append(prediction)

    # Obliczenie macierzy konfuzji
    labels = ["spam", "ham"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_percent = cm / np.sum(cm) * 100
    acc = accuracy_score(y_true, y_pred) * 100

    print("\n📊 MACIERZ KONFUZJI (%):")
    print(f"      spam      ham")
    print(f"spam  {cm_percent[0,0]:6.2f}%   {cm_percent[0,1]:6.2f}%")
    print(f"ham   {cm_percent[1,0]:6.2f}%   {cm_percent[1,1]:6.2f}%")
    print(f"\n🎯 DOKŁADNOŚĆ (accuracy): {acc:.2f}%")

if __name__ == "__main__":
    main()
