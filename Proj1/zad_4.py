import os
import string
import random
import time
from email import message_from_file
from collections import Counter, defaultdict

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

from datasketch import MinHash, MinHashLSH
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np

# === KONFIGURACJA ===
INDEX_PATH = "trec07p/full/index"
DATA_PATH = "trec07p"
TRAIN_RATIO = 0.8
SAMPLE_SIZE = 100  # np. 2000 dla szybkiego testu, None = całość
RESULTS_FILE = "results_lsh.txt"

# MinHash / Shingle params
NUM_PERM = 128
SHINGLE_SIZE = 3  # k for k-shingles (on tokens)
USE_STEMMING = True  # możesz ustawić False aby porównać wpływ stemizacji tutaj
THRESHOLDS = [0.1, 0.3, 0.5, 0.7, 0.9]  # testowane progi LSH

# Domyślna etykieta gdy brak dopasowań (często ham)
DEFAULT_LABEL = "ham"

random.seed(42)


# === POMOCNICZE FUNKCJE ===
def load_index(index_path):
    entries = []
    with open(index_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                label, path = parts[0], parts[1]
                # Normalizuj ścieżkę: '../data/inmail.X' -> 'trec07p/data/inmail.X'
                full_path = os.path.join(DATA_PATH, path.replace("../", ""))
                entries.append((full_path, label))
    return entries


def load_email_content(filepath):
    """Wczytuje zawartość e-maila i zwraca string tekstowy (ignoruje błędy kodowania)."""
    try:
        with open(filepath, "r", encoding="latin-1") as f:
            msg = message_from_file(f)
            payload = ""
            if msg.is_multipart():
                # złącz wszystkie części tekstowe
                parts = []
                for part in msg.walk():
                    # tylko tekstowe części (ignore attachments)
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
            return payload or ""
    except Exception:
        return ""


def preprocess_text(text, use_stemming=True):
    """lower, remove punctuation, tokenize, remove stopwords, optional stemming; zwraca listę tokenów."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    sw = set(stopwords.words("english"))
    tokens = [t for t in tokens if t.isalpha() and t not in sw]
    if use_stemming:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(t) for t in tokens]
    return tokens


def get_shingles(tokens, k=3):
    """Zwraca listę shingli (k-gramów) utworzonych z tokenów (continuous k-word shingles)."""
    if len(tokens) < k:
        # fallback: użyj pojedynczych tokenów
        return tokens
    shingles = []
    for i in range(len(tokens) - k + 1):
        sh = " ".join(tokens[i:i + k])
        shingles.append(sh)
    return shingles


def build_minhash_from_shingles(shingles, num_perm=128):
    """Zwraca MinHash obliczony na zestawie shingles (unikatowych)."""
    m = MinHash(num_perm=num_perm)
    # używamy zestawu, aby uniknąć wielokrotnego dodawania tego samego shingla
    for s in set(shingles):
        m.update(s.encode("utf8"))
    return m


# === PROCEDURY TRENING / TEST ===
def prepare_train_min_hashes(train_entries, use_stemming=True, shingle_k=3, num_perm=128):
    """
    Przygotowuje MinHash dla każdego dokumentu treningowego i mapuje identyfikatory na etykiety.
    Zwraca:
      - dict id -> MinHash
      - dict id -> label
    """
    id_to_minhash = {}
    id_to_label = {}
    for idx, (path, label) in enumerate(train_entries):
        text = load_email_content(path)
        tokens = preprocess_text(text, use_stemming)
        shingles = get_shingles(tokens, k=shingle_k)
        m = build_minhash_from_shingles(shingles, num_perm=num_perm)
        doc_id = f"doc{idx}"
        id_to_minhash[doc_id] = m
        id_to_label[doc_id] = label
    return id_to_minhash, id_to_label


def classify_with_lsh(lsh, train_label_map, test_entries, use_stemming=True, shingle_k=3, num_perm=128):
    """
    Dla każdego dokumentu testowego:
      - oblicz MinHash
      - zapytaj lsh.query(minhash) -> listę pasujących doc_id
      - jeśli lista niepusta: dokonaj głosowania etykiet (majority vote)
      - jeśli pusta: przypisz DEFAULT_LABEL
    Zwraca y_true, y_pred
    """
    y_true = []
    y_pred = []
    for path, label in test_entries:
        text = load_email_content(path)
        tokens = preprocess_text(text, use_stemming)
        shingles = get_shingles(tokens, k=shingle_k)
        m = build_minhash_from_shingles(shingles, num_perm=num_perm)
        matches = lsh.query(m)  # list of doc_ids
        if matches:
            # majority vote among labels of matches
            votes = [train_label_map[mid] for mid in matches if mid in train_label_map]
            if votes:
                counter = Counter(votes)
                pred = counter.most_common(1)[0][0]
            else:
                pred = DEFAULT_LABEL
        else:
            pred = DEFAULT_LABEL
        y_true.append(label)
        y_pred.append(pred)
    return y_true, y_pred


# === GŁÓWNY PROGRAM ===
def main():
    print("📂 Wczytywanie indexu i danych...")
    index_entries = load_index(INDEX_PATH)
    random.shuffle(index_entries)

    if SAMPLE_SIZE:
        index_entries = index_entries[:SAMPLE_SIZE]
        print(f"⚠️ SAMPLE_SIZE active: using first {len(index_entries)} entries")

    split_point = int(len(index_entries) * TRAIN_RATIO)
    train_entries = index_entries[:split_point]
    test_entries = index_entries[split_point:]

    print(f"Łącznie: {len(index_entries)} dokumentów; trening: {len(train_entries)}; test: {len(test_entries)}")
    results_lines = []
    results_lines.append(f"LSH MinHash results\nSAMPLE_SIZE={SAMPLE_SIZE}\nNUM_PERM={NUM_PERM}\nSHINGLE_SIZE={SHINGLE_SIZE}\nUSE_STEMMING={USE_STEMMING}\n")

    # Przygotuj MinHash na treningu (raz) - będziemy je ponownie wstawiać do nowych LSH dla różnych thresholdów.
    print("🧠 Budowanie MinHash dla zbioru treningowego...")
    t0 = time.time()
    train_mh_map, train_label_map = prepare_train_min_hashes(train_entries, use_stemming=USE_STEMMING,
                                                            shingle_k=SHINGLE_SIZE, num_perm=NUM_PERM)
    t_prep = time.time() - t0
    print(f"Gotowe. Czas przygotowania MinHash treningu: {t_prep:.2f}s")
    results_lines.append(f"prepare_time={t_prep:.2f}s\n")

    # Dla każdego threshold budujemy nowy MinHashLSH (z tym samym num_perm) i wstawiamy minhashy treningowe.
    for thresh in THRESHOLDS:
        print(f"\n🔎 Test dla threshold = {thresh}")
        results_lines.append(f"\nTHRESHOLD={thresh}\n")
        # buduj LSH z parametrem threshold
        t0 = time.time()
        lsh = MinHashLSH(threshold=thresh, num_perm=NUM_PERM)
        # insert training minhashes
        for doc_id, mh in train_mh_map.items():
            lsh.insert(doc_id, mh)
        build_time = time.time() - t0
        print(f"LSH zbudowano w {build_time:.2f}s")

        # klasyfikacja testów
        t1 = time.time()
        y_true, y_pred = classify_with_lsh(lsh, train_label_map, test_entries,
                                          use_stemming=USE_STEMMING, shingle_k=SHINGLE_SIZE, num_perm=NUM_PERM)
        elapsed = time.time() - t1

        # metryki
        labels = ["spam", "ham"]
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        cm_percent = cm / np.sum(cm) * 100
        acc = accuracy_score(y_true, y_pred) * 100

        # raport w konsoli
        print(f"🎯 Accuracy: {acc:.2f}% | build_time: {build_time:.2f}s | classify_time: {elapsed:.2f}s")
        print("📊 Confusion matrix (%):")
        print(f"      spam      ham")
        print(f"spam  {cm_percent[0,0]:6.2f}%   {cm_percent[0,1]:6.2f}%")
        print(f"ham   {cm_percent[1,0]:6.2f}%   {cm_percent[1,1]:6.2f}%")

        # zapisz wyniki
        results_lines.append(f"accuracy={acc:.2f}%\n")
        results_lines.append(f"build_time={build_time:.2f}s classify_time={elapsed:.2f}s\n")
        results_lines.append("confusion_percent:\n")
        results_lines.append(f"spam_spam={cm_percent[0,0]:6.2f}% spam_ham={cm_percent[0,1]:6.2f}%\n")
        results_lines.append(f"ham_spam={cm_percent[1,0]:6.2f}% ham_ham={cm_percent[1,1]:6.2f}%\n")

    # zapis do pliku
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(results_lines))

    print(f"\n📁 Wyniki zapisano do: {RESULTS_FILE}")
    print("✅ Gotowe.")


if __name__ == "__main__":
    main()