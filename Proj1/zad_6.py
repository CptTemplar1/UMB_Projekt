import os
import string
import random
import time
from email import message_from_file

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# === KONFIGURACJA ===
INDEX_PATH = "trec07p/full/index"   # ścieżka do indexu
DATA_PATH = "trec07p"               # ścieżka do danych
TRAIN_RATIO = 0.8                   # stosunek danych treningowych do testowych
SAMPLE_SIZE = None                  # ograniczenie liczby próbek, np. 2000 dla testów, None = całość
MAX_FEATURES = 20000                # rozmiar wektora TF-IDF (zmniejsz do 5000 jeśli brakuje pamięci)
SAMPLE_SEED = 42                    # ustawienie ziarna losowości

EPOCHS = 5                          # liczba epok treningu 
BATCH_SIZE = 128                    # rozmiar batcha
RESULTS_FILE = "results_dnn.txt"    # nazwa pliku wynikowego

USE_PREPROCESSING = True            # Czy użyć preprocessingu NLTK (stopwords + stemming) przed TF-IDF

# Modele do przetestowania: lista dictów (nazwa, architektura, activation_hidden)
MODEL_CONFIGS = [
    {"name": "small", "layers": [64], "activation": "relu"},
    {"name": "medium", "layers": [128, 64], "activation": "relu"},
    {"name": "large", "layers": [256, 128, 64], "activation": "relu"},
    {"name": "small_tanh", "layers": [64], "activation": "tanh"},
]

# Ustawienie ziarna losowości dla powtarzalności
random.seed(SAMPLE_SEED)
np.random.seed(SAMPLE_SEED)
tf.random.set_seed(SAMPLE_SEED)


# === POMOCNICZE FUNKCJE ===
# Wczytuje indeks plików e-maili i ich etykiety (spam/ham)
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

# Wczytuje treść e-maila (temat + ciało) jako zwykły tekst. Ignoruje błędy kodowania.
def load_email_content(filepath):
    try:
        with open(filepath, "r", encoding="latin-1") as f:
            msg = message_from_file(f)
            subject = msg.get("Subject", "") or ""
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
            return (subject + " " + payload).strip()
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


# Wczytuje teksty i etykiety, przeprowadza opcjonalny preprocessing, zwraca listę tekstów i tablicę etykiet (spam/ham)
def prepare_corpus(entries, use_preprocessing=True, sample_size=None):
    texts = []
    labels = []
    count = 0
    for path, label in entries:
        if sample_size and count >= sample_size:
            break
        txt = load_email_content(path)
        if use_preprocessing:
            txt = preprocess_text(txt)
        texts.append(txt)
        labels.append(1 if label == "spam" else 0)  # spam=1, ham=0
        count += 1
    return texts, np.array(labels)


# Tworzy i dopasowuje wektorizer TF-IDF, zwraca wektorizer i macierz cech
def build_vectorizer(texts, max_features=20000):
    vec = TfidfVectorizer(max_features=max_features, ngram_range=(1,2))
    X = vec.fit_transform(texts)
    return vec, X


# Buduje model DNN według podanej architektury 
def build_model(input_dim, layer_sizes, activation_hidden="relu", dropout=0.2, lr=1e-3):
    model = Sequential()
    # Warstwa wejściowa jest częścią pierwszej warstwy ukrytej
    for i, size in enumerate(layer_sizes):
        if i == 0:
            model.add(Dense(size, activation=activation_hidden, input_shape=(input_dim,)))
        else:
            model.add(Dense(size, activation=activation_hidden))
        model.add(Dropout(dropout))
    # Warstwa wyjściowa - sigmoid dla binarnej klasyfikacji
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer=Adam(learning_rate=lr),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model


# === GŁÓWNY PROGRAM ===
def main():
    print("📂 Wczytywanie danych...")
    entries = load_index(INDEX_PATH)
    random.shuffle(entries)

    if SAMPLE_SIZE:
        use_entries = entries[:SAMPLE_SIZE]
        print(f"⚠️ SAMPLE_SIZE aktywne. Wykorzystuję {len(use_entries)} pierwszych wpisów.")
    else:
        use_entries = entries

    # Przygotowanie tekstów i etykiet
    print("🧾 Przygotowanie korpusu tekstów (preprocessing = %s)..." % USE_PREPROCESSING)
    texts, labels = prepare_corpus(use_entries, use_preprocessing=USE_PREPROCESSING, sample_size=None)
    print(f"Przygotowano {len(texts)} dokumentów.")

    # Podział na trening/test (z zachowaniem TRAIN_RATIO)
    split_point = int(len(texts) * TRAIN_RATIO)
    X_texts_train = texts[:split_point]
    X_texts_test = texts[split_point:]
    y_train = labels[:split_point]
    y_test = labels[split_point:]
    print(f"Trening: {len(X_texts_train)}, Test: {len(X_texts_test)}")

    # Tworzenie wektorów TF-IDF
    print(f"🔤 Tworzenie TF-IDF (max_features={MAX_FEATURES})...")
    vectorizer, X_train_sparse = build_vectorizer(X_texts_train, max_features=MAX_FEATURES)
    X_test_sparse = vectorizer.transform(X_texts_test)

    # Konwersja do dense (Keras wymaga gęstych (Dense) macierzy)
    print("Konwersja do macierzy gęstych...")
    X_train = X_train_sparse.toarray().astype(np.float32)
    X_test = X_test_sparse.toarray().astype(np.float32)
    input_dim = X_train.shape[1]
    print(f"Input dim = {input_dim}")

    results_lines = []
    results_lines.append(f"DNN TF-IDF results\nSAMPLE_SIZE={SAMPLE_SIZE}\nMAX_FEATURES={MAX_FEATURES}\nEPOCHS={EPOCHS}\nBATCH_SIZE={BATCH_SIZE}\nUSE_PREPROCESSING={USE_PREPROCESSING}\n\n")

    # Dla każdej konfiguracji modelu trenuje, testuje i zapisuje wyniki
    for cfg in MODEL_CONFIGS:
        name = cfg["name"]
        layers = cfg["layers"]
        activation = cfg.get("activation", "relu")
        print(f"\n=== Model: {name} | layers={layers} | activation={activation} ===")
        model = build_model(input_dim=input_dim, layer_sizes=layers, activation_hidden=activation)

        # Trening modelu
        t0 = time.time()
        history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
        train_time = time.time() - t0
        print(f"Trening zakończony w {train_time:.2f}s")

        # Predykcja na zbiorze testowym
        t1 = time.time()
        y_prob = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0).ravel()
        y_pred = (y_prob >= 0.5).astype(int)
        predict_time = time.time() - t1

        # Metryki ewaluacyjne
        acc = accuracy_score(y_test, y_pred) * 100.0
        labels_order = [1, 0]  # spam=1, ham=0
        cm = confusion_matrix(y_test, y_pred, labels=labels_order)
        cm_percent = cm / np.sum(cm) * 100.0

        # Wypisuje wyniki i zapisuje je do pliku
        print(f"🎯 Accuracy: {acc:.2f}% | Czas treningu: {train_time:.2f}s | Czas predykcji: {predict_time:.2f}s")
        print("📊 Confusion matrix (%):")
        print("      spam      ham")
        print(f"spam  {cm_percent[0,0]:6.2f}%   {cm_percent[0,1]:6.2f}%")
        print(f"ham   {cm_percent[1,0]:6.2f}%   {cm_percent[1,1]:6.2f}%")

        results_lines.append(f"Model: {name}\n")
        results_lines.append(f"layers={layers} activation={activation}\n")
        results_lines.append(f"accuracy={acc:.2f}% train_time={train_time:.2f}s predict_time={predict_time:.2f}s\n")
        results_lines.append(f"confusion_percent:\nspam_spam={cm_percent[0,0]:6.2f}% spam_ham={cm_percent[0,1]:6.2f}%\n")
        results_lines.append(f"ham_spam={cm_percent[1,0]:6.2f}% ham_ham={cm_percent[1,1]:6.2f}%\n\n")

        # Zwolnij pamięć modelu przed kolejnym testem
        tf.keras.backend.clear_session()

    # Zapis do pliku wyników
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(results_lines))
    
    print(f"\n📁 Wyniki zapisano do: {RESULTS_FILE}")

if __name__ == "__main__":
    main()