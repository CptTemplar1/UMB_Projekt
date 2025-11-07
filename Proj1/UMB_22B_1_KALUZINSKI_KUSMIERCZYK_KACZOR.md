# Uczenie Maszynowe w Bezpieczeństwie
## Projekt 1
### Grupa 22B
### Autorzy: Przemysław Kałużiński, Jakub Kuśmierczyk, Michał Kaczor

### Zadanie 1
Pobrać, rozpakować i przeanalizować strukturę plików i katalogów archiwum zawierającego wiadomości poczty elektronicznej.  
Dane te dostępne są pod adresem:  
https://plg.uwaterloo.ca/~gvcormac/treccorpus07/  
**Uwaga.** Nie należy otwierać plików z archiwum ani w przeglądarce HTML ani w programie pocztowym!

#### Wyniki

Pomyślnie pobrano i rozpakowano archiwum `TREC 2007 Public Corpus`, które będzie wykorzystywane w dalszych zadaniach projektu. Ze względu na rozmiar archiwum (około 450 MB) oraz potencjalnie niebezpieczną zawartość wiadomości spam (które mogą zawierać złośliwe linki lub nieodpowiednie treści), archiwum zostało wykluczone z repozytorium Git poprzez dodanie do pliku `.gitignore`.

**Archiwum TREC 2007**  
TREC 2007 Public Corpus to publicznie dostępne archiwum wiadomości email używane do badań nad filtrowaniem spamu. Zbiór został opracowany w ramach Text Retrieval Conference (TREC) i stanowi standardowy benchmark do testowania algorytmów klasyfikacji wiadomości email.

Archiwum posiada następującą strukturę katalogów:  
trec07p/  
├── data/ - Główny folder z wiadomościami  
├── full/ - Folder z pełnym indeksem  
├── delay/ - Dane feedback tylko dla pierwszych 10,000 wiadomości  
└── partial/ - Dane feedback tylko dla 30,388 wiadomości odpowiadających 1 odbiorcy

**Folder `data/`**:
- Zawiera 75,419 wiadomości email w postaci plików tekstowych
- Pliki mają nazwy w formacie `inmail.X`, gdzie X to liczba od 1 do 75419
- Każdy plik zawiera pełną wiadomość email w formacie MIME

**Folder `full/`**:
- Zawiera plik `index` będący słownikiem klasyfikacji
- Format wpisów: `[etykieta] [ścieżka_do_pliku]`, np. `spam ../data/inmail.1`
- Etykiety: "spam" (niechciane wiadomości) lub "ham" (pożądane wiadomości)

**Foldery dodatkowe (nieużywane w projekcie)**:
- `delay/` - zawiera dane feedback tylko dla pierwszych 10,000 wiadomości
- `partial/` - zawiera dane feedback tylko dla 30,388 wiadomości odpowiadających jednemu odbiorcy

---

**Statystyki zbioru danych**:
- **Łączna liczba wiadomości**: 75,419
- **Wiadomości ham (pożądane)**: 25,220 (33.4%)
- **Wiadomości spam (niechciane)**: 50,199 (66.6%)
- **Rozkład**: Przewaga wiadomości spam

### Zadanie 2
Wykorzystując informacje z wykładu oraz stosując technikę zakazanych słów kluczowych (blacklist), dokonać klasyfikacji binarnej wiadomości z archiwum z podziałem na: spam (wiadomości typu spam) oraz ham (wiadomości pożądane).

**Uwagi:**
1. Przed przystąpieniem do procesu klasyfikacji usunąć z wiadomości stopping words (np. the, is, are, . . . ),
dokonać stemizacji słów w wiadomościach oraz ekstrakcji tokenów.
2. Do realizacji zadania użyć języka Python oraz bibliotek: string, email, NLTK, os.
3. Zbiór zakazanych słów kluczowych powinien być wygenerowany na podstawie danych z podzbioru treningowego,
natomiast ewaluacja danych uzyskanych z podzbioru testowego.
4. Wynikiem ewaluacji powinna być macierz konfuzji (procentowa) oraz wartość wskaźnika accuracy, również w
postaci procentowej.

#### Implementacja

Ze względu na fakt, że kod implementujący zadania 2 i 3 jest ze sobą ściśle powiązany, to pełna implementacja obu zadań została umieszczona w rozdziale **implementacja** zadania 3. 

#### Wyniki

Podobnie jak implementacja, wyniki obu zadań 2 i 3 zostały przedstawione w rozdziale **wyniki** zadania 3, ponieważ kod programu zwraca wyniki obu zadań jednocześnie.

### Zadanie 3
Zweryfikować wpływ stemizacji na pracę algorytmu zadania drugiego a następnie porównać uzyskane wyniki.

#### Implementacja

**1. Konfiguracja globalna**

Na wstępie programu znajduje się kod, który definiuje stałe konfiguracyjne używane w całym programie. Ułatwia to dostosowanie parametrów bez konieczności modyfikowania logiki programu.

**Kod:**
``` python
INDEX_PATH = "trec07p/full/index"       # ścieżka do indexu
DATA_PATH = "trec07p"                   # ścieżka do danych
TRAIN_RATIO = 0.8                       # stosunek danych treningowych do testowych
TOP_N = 100                             # liczba słów w blacklist
SAMPLE_SIZE = None                      # ograniczenie liczby próbek, np. 2000 dla testów, None = całość
RESULTS_FILE = "results_stemming.txt"   # nazwa pliku wynikowego
```

**2. Funkcja `load_index`**

**Wejście:**  
- `index_path` (string) - ścieżka do pliku z indeksem wiadomości

**Wyjście:**  
- `entries` (list) - lista krotek zawierających pełną ścieżkę do pliku i etykietę (spam/ham)

**Opis:**  
Funkcja wczytuje plik indeksu, gdzie każda linia zawiera etykietę (spam/ham) i ścieżkę do pliku z wiadomością. Parsuje każdą linię, tworzy pełną ścieżkę do pliku (usuwając "../" z oryginalnej ścieżki) i zwraca listę wszystkich wpisów.

**Kod:**
``` python
def load_index(index_path):
    entries = []
    with open(index_path, "r") as f:
        for line in f:
            label, path = line.strip().split()
            full_path = os.path.join(DATA_PATH, path.replace("../", ""))
            entries.append((full_path, label))
    return entries
```

---

**3. Funkcja `preprocess_text`**

**Wejście:**  
- `text` (string) - tekst wiadomości email do przetworzenia
- `use_stemming` (bool) - flaga określająca czy stosować stemizację

**Wyjście:**  
- `tokens` (list) - lista przetworzonych tokenów (słów)

**Opis:**  
Funkcja przeprowadza pełne przetwarzanie tekstu: konwersja na małe litery, usuwanie znaków interpunkcyjnych, tokenizacja na pojedyncze słowa, usuwanie stopwords (słów bez znaczenia) oraz opcjonalna stemizacja przy użyciu algorytmu PorterStemmer. Zwraca listę oczyszczonych tokenów.

**Kod:**
``` python
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
```

---

**4. Funkcja `load_email_content`**

**Wejście:**  
- `filepath` (string) - ścieżka do pliku z wiadomością email

**Wyjście:**  
- `text` (string) - wyekstrahowana treść wiadomości lub pusty string w przypadku błędu

**Opis:**  
Funkcja wczytuje i parsuje wiadomość email przy użyciu biblioteki email. Obsługuje wiadomości wieloczęściowe (multipart), dekoduje zawartość i zwraca czysty tekst wiadomości. W przypadku błędów zwraca pusty string.

**Kod:**
``` python
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
```

---

**5. Funkcja `build_blacklist`**

**Wejście:**  
- `train_data` (list) - lista krotek (tokens, label) z danych treningowych
- `top_n` (int) - liczba słów do umieszczenia na blackliście

**Wyjście:**  
- `blacklist` (list) - lista słów kluczowych najbardziej charakterystycznych dla spamu

**Opis:**  
Funkcja analizuje dane treningowe, zliczając wystąpienia słów w spamie i hamie. Dla każdego słowa oblicza stosunek częstotliwości w spamie do częstotliwości w hamie. Zwraca listę `top_n` słów z najwyższym stosunkiem, które będą używane jako zakazane słowa kluczowe.

**Kod:**
``` python
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
```

---

**6. Funkcja `classify_email`**

**Wejście:**  
- `tokens` (list) - lista tokenów z przetworzonej wiadomości
- `blacklist` (list) - lista zakazanych słów kluczowych

**Wyjście:**  
- `"spam"` lub `"ham"` (string) - wynik klasyfikacji

**Opis:**  
Funkcja klasyfikuje wiadomość jako spam, jeśli którykolwiek z tokenów znajduje się na blackliście. W przeciwnym przypadku klasyfikuje jako ham. Jest to prosty klasyfikator oparty na zasadzie "czarnej listy".

**Kod:**  
``` python
def classify_email(tokens, blacklist):
    return "spam" if any(word in blacklist for word in tokens) else "ham"
```

---

**7. Funkcja `evaluate_model`**

**Wejście:**  
- `train_entries` (list) - lista krotek (ścieżka, etykieta) dla danych treningowych
- `test_entries` (list) - lista krotek (ścieżka, etykieta) dla danych testowych
- `use_stemming` (bool) - flaga określająca czy stosować stemizację

**Wyjście:**  
- `acc` (float) - dokładność klasyfikacji w procentach
- `cm_percent` (numpy.ndarray) - macierz konfuzji w procentach
- `elapsed` (float) - czas wykonania w sekundach

**Opis:**  
Funkcja przeprowadza pełny proces uczenia i ewaluacji modelu: przetwarza dane treningowe, buduje blacklistę, klasyfikuje wiadomości testowe i oblicza metryki wydajności. Zwraca accuracy, macierz konfuzji i czas wykonania.

**Kod:**
``` python
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
```

---

**8. Funkcja `main`**

**Wejście:**  
- Brak parametrów wejściowych

**Wyjście:**  
- Brak bezpośredniego wyjścia (funkcja wykonuje program i zapisuje wyniki do pliku)

**Opis:**  
Główna funkcja programu, która koordynuje cały proces: wczytuje i tasuje dane, dzieli na zbiór treningowy i testowy, przeprowadza dwa eksperymenty (ze stemizacją i bez), porównuje wyniki, wyświetla raport i zapisuje wyniki do pliku tekstowego.

**Kod:**
``` python
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
    print("🧠 Test 1: ZE STEMIZACJĄ")
    acc_stem, cm_stem, time_stem = evaluate_model(train_entries, test_entries, use_stemming=True)
    print(f"🎯 Accuracy (stem): {acc_stem:.2f}% | ⏱ Czas: {time_stem:.2f}s")
    results_log.append(f"Test 1 (ze stemizacją): accuracy={acc_stem:.2f}%, czas={time_stem:.2f}s")

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
        f"Ze stemizacją:    {acc_stem:.2f}% ({time_stem:.2f}s)\n"
        f"Bez stemizacji:  {acc_no_stem:.2f}% ({time_no_stem:.2f}s)\n"
        f"🧩 Różnica dokładności: {diff_acc:+.2f}%\n"
        f"⏱ Różnica czasu: {diff_time:+.2f}s (wartość dodatnia = wolniej ze stemizacją)\n"
    )

    print(summary)
    results_log.append(summary)

    # Macierze konfuzji
    matrix_report = (
        "\n📊 MACIERZ KONFUZJI (ZE STEMIZACJĄ):\n"
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
```

---

**9. Kompletny kod**  
Poniżej znajduje się kompletny kod programu, który można uruchomić.

**Kod:**
``` python
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
INDEX_PATH = "trec07p/full/index"       # ścieżka do indexu
DATA_PATH = "trec07p"                   # ścieżka do danych
TRAIN_RATIO = 0.8                       # stosunek danych treningowych do testowych
TOP_N = 100                             # liczba słów w blacklist
SAMPLE_SIZE = None                      # ograniczenie liczby próbek, np. 2000 dla testów, None = całość
RESULTS_FILE = "results_stemming.txt"   # nazwa pliku wynikowego


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
    print("🧠 Test 1: ZE STEMIZACJĄ")
    acc_stem, cm_stem, time_stem = evaluate_model(train_entries, test_entries, use_stemming=True)
    print(f"🎯 Accuracy (stem): {acc_stem:.2f}% | ⏱ Czas: {time_stem:.2f}s")
    results_log.append(f"Test 1 (ze stemizacją): accuracy={acc_stem:.2f}%, czas={time_stem:.2f}s")

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
        f"ZE stemizacją:    {acc_stem:.2f}% ({time_stem:.2f}s)\n"
        f"Bez stemizacji:  {acc_no_stem:.2f}% ({time_no_stem:.2f}s)\n"
        f"🧩 Różnica dokładności: {diff_acc:+.2f}%\n"
        f"⏱ Różnica czasu: {diff_time:+.2f}s (wartość dodatnia = wolniej ze stemizacją)\n"
    )

    print(summary)
    results_log.append(summary)

    # Macierze konfuzji
    matrix_report = (
        "\n📊 MACIERZ KONFUZJI (ZE STEMIZACJĄ):\n"
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
```

#### Wyniki

```text
📂 Wczytywanie danych...
🧠 Test 1: ZE STEMIZACJĄ
🎯 Accuracy (stem): 61.83% | ⏱ Czas: 2465.29s

🧠 Test 2: BEZ STEMIZACJI
🎯 Accuracy (no stem): 58.64% | ⏱ Czas: 239.89s

📊 PORÓWNANIE WYNIKÓW
Ze stemizacją:    61.83% (2465.29s)
Bez stemizacji:  58.64% (239.89s)
🧩 Różnica dokładności: +3.20%
⏱ Różnica czasu: +2225.39s (wartość dodatnia = wolniej ze stemizacją)


📊 MACIERZ KONFUZJI (ZE STEMIZACJĄ):
      spam      ham
spam  28.65%    38.07%
ham   0.09%     33.18%

📊 MACIERZ KONFUZJI (BEZ STEMIZACJI):
      spam      ham
spam  25.45%    41.28%
ham   0.09%     33.19%

📁 Wyniki zapisano do pliku: results_stemming.txt
```

### Zadanie 4
Dokonać klasyfikacji binarnej wiadomości z archiwum (zadanie 1) na spam i ham, stosując algorytmy rozmytego haszowania.

**Uwagi:**
1. Do tego celu użyć algorytmu LSH (MinHash, MinHashLSH) z biblioteki datasketch.
2. Wyniki pracy algorytmu przedstawić przy pomocy procentowej macierzy konfuzji i wskaźnika accuracy.
3. Sprawdzić pracę programu dla różnych wartości parametru threshold funkcji MinHashLSH.
4. Porównać uzyskane wyniki z wynikami z poprzednich zadań.

#### Implementacja

**1. Konfiguracja globalna**

Na wstępie programu znajduje się kod, który definiuje stałe konfiguracyjne używane w całym programie. Ułatwia to dostosowanie parametrów bez konieczności modyfikowania logiki programu.

**Kod:**  
``` python
INDEX_PATH = "trec07p/full/index"       # ścieżka do indexu
DATA_PATH = "trec07p"                   # ścieżka do danych
TRAIN_RATIO = 0.8                       # stosunek danych treningowych do testowych
SAMPLE_SIZE = None                      # ograniczenie liczby próbek, np. 2000 dla testów, None = całość
RESULTS_FILE = "results_lsh.txt"        # nazwa pliku wynikowego

# Parametry LSH / MinHash
NUM_PERM = 128                          # liczba permutacji w MinHash
SHINGLE_SIZE = 3                        # rozmiar shingli (k-gramów)
USE_STEMMING = True                     # czy stosować stemizację
THRESHOLDS = [0.1, 0.3, 0.5, 0.7, 0.9]  # testowane progi LSH
DEFAULT_LABEL = "ham"                   # etykieta domyślna, gdy brak dopasowań w LSH

random.seed(42)                         # ustawienie ziarna losowości
```

**2. Funkcja `load_index`**

**Wejście:**  
- `index_path` (string) - ścieżka do pliku z indeksem wiadomości

**Wyjście:**  
- `entries` (list) - lista krotek zawierających pełną ścieżkę do pliku i etykietę (spam/ham)

**Opis:**  
Funkcja wczytuje plik indeksu, parsuje każdą linię rozdzielając ją na etykietę i ścieżkę do pliku. Normalizuje ścieżki usuwając "../" i tworzy pełne ścieżki względem głównego katalogu danych. Zwraca listę wszystkich wpisów gotowych do dalszego przetwarzania.

**Kod:**  
``` python
def load_index(index_path):
    entries = []
    with open(index_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                label, path = parts[0], parts[1]
                # Normalizuje ścieżkę: '../data/inmail.X' -> 'trec07p/data/inmail.X'
                full_path = os.path.join(DATA_PATH, path.replace("../", ""))
                entries.append((full_path, label))
    return entries
```

---

**3. Funkcja `load_email_content`**

**Wejście:**  
- `filepath` (string) - ścieżka do pliku z wiadomością email

**Wyjście:**  
- `payload` (string) - wyekstrahowana treść wiadomości lub pusty string w przypadku błędu

**Opis:**  
Funkcja wczytuje i parsuje wiadomość email przy użyciu biblioteki email. Obsługuje zarówno wiadomości wieloczęściowe (multipart) jak i pojedyncze. Dla wiadomości wieloczęściowych iteruje przez wszystkie części i wyciąga tylko te o typie tekstowym. Dekoduje zawartość binarną i obsługuje błędy kodowania. W przypadku wyjątków zwraca pusty string.

**Kod:**  
``` python
def load_email_content(filepath):
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
```

---

**4. Funkcja `preprocess_text`**

**Wejście:**  
- `text` (string) - tekst wiadomości email do przetworzenia
- `use_stemming` (bool) - flaga określająca czy stosować stemizację

**Wyjście:**  
- `tokens` (list) - lista przetworzonych tokenów (słów)

**Opis:**  
Funkcja przeprowadza pełne przetwarzanie tekstu przed użyciem w algorytmie LSH: konwersja na małe litery, usuwanie znaków interpunkcyjnych, tokenizacja na pojedyncze słowa, filtrowanie tylko słów alfabetycznych, usuwanie stopwords (słów bez znaczenia) oraz opcjonalna stemizacja przy użyciu algorytmu PorterStemmer.

**Kod:**  
``` python
def preprocess_text(text, use_stemming=True):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    sw = set(stopwords.words("english"))
    tokens = [t for t in tokens if t.isalpha() and t not in sw]
    if use_stemming:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(t) for t in tokens]
    return tokens
```

---

**5. Funkcja `get_shingles`**

**Wejście:**  
- `tokens` (list) - lista tokenów z przetworzonej wiadomości
- `k` (int) - rozmiar shingli (k-gramów)

**Wyjście:**  
- `shingles` (list) - lista shingli utworzonych z ciągłych sekwencji tokenów

**Opis:**  
Funkcja tworzy k-gramy (shingle) z ciągłych sekwencji tokenów. Dla podanej listy tokenów tworzy wszystkie możliwe ciągłe sekwencje o długości k, łącząc je w stringi. Jeśli długość tokenów jest mniejsza niż k, zwraca oryginalne tokeny jako fallback.

**Kod:**  
``` python
def get_shingles(tokens, k=3):
    if len(tokens) < k:
        # fallback: użyj pojedynczych tokenów
        return tokens
    shingles = []
    for i in range(len(tokens) - k + 1):
        sh = " ".join(tokens[i:i + k])
        shingles.append(sh)
    return shingles
```

---

**6. Funkcja `build_minhash_from_shingles`**

**Wejście:**  
- `shingles` (list) - lista shingli (k-gramów)
- `num_perm` (int) - liczba permutacji dla algorytmu MinHash

**Wyjście:**  
- `m` (MinHash) - obiekt MinHash reprezentujący dokument

**Opis:**  
Funkcja tworzy obiekt MinHash dla dokumentu na podstawie jego shingli. Używa zestawu unikalnych shingli aby uniknąć duplikatów. Każdy shingle jest kodowany do postaci bajtów przed dodaniem do MinHash. Parametr num_perm określa dokładność haszowania.

**Kod:**  
``` python
def build_minhash_from_shingles(shingles, num_perm=128):
    m = MinHash(num_perm=num_perm)
    # używamy zestawu, aby uniknąć wielokrotnego dodawania tego samego shingla
    for s in set(shingles):
        m.update(s.encode("utf8"))
    return m
```

---

**7. Funkcja `prepare_train_min_hashes`**

**Wejście:**  
- `train_entries` (list) - lista krotek (ścieżka, etykieta) dla danych treningowych
- `use_stemming` (bool) - flaga określająca czy stosować stemizację
- `shingle_k` (int) - rozmiar shingli
- `num_perm` (int) - liczba permutacji dla MinHash

**Wyjście:**  
- `id_to_minhash` (dict) - słownik mapujący ID dokumentu na jego MinHash
- `id_to_label` (dict) - słownik mapujący ID dokumentu na jego etykietę

**Opis:**  
Funkcja przetwarza wszystkie dokumenty treningowe: wczytuje treść, przetwarza tekst, tworzy shingle, buduje MinHash dla każdego dokumentu. Przypisuje unikalne ID każdemu dokumentowi i zwraca dwa słowniki do dalszego użycia w LSH.

**Kod:**  
``` python
def prepare_train_min_hashes(train_entries, use_stemming=True, shingle_k=3, num_perm=128):
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
```

---

**8. Funkcja `classify_with_lsh`**

**Wejście:**  
- `lsh` (MinHashLSH) - obiekt LSH z wstawionymi dokumentami treningowymi
- `train_label_map` (dict) - słownik mapujący ID na etykiety treningowe
- `test_entries` (list) - lista krotek (ścieżka, etykieta) dla danych testowych
- `use_stemming` (bool) - flaga określająca czy stosować stemizację
- `shingle_k` (int) - rozmiar shingli
- `num_perm` (int) - liczba permutacji dla MinHash

**Wyjście:**  
- `y_true` (list) - lista prawdziwych etykiet
- `y_pred` (list) - lista przewidywanych etykiet

**Opis:**  
Funkcja klasyfikuje dokumenty testowe używając LSH. Dla każdego dokumentu testowego: przetwarza tekst, tworzy shingle, buduje MinHash, pyta LSH o podobne dokumenty. Jeśli znaleziono dopasowania, przeprowadza głosowanie większościowe na podstawie etykiet dokumentów treningowych. Jeśli brak dopasowań, używa etykiety domyślnej.

**Kod:**  
``` python
def classify_with_lsh(lsh, train_label_map, test_entries, use_stemming=True, shingle_k=3, num_perm=128):
    y_true = []
    y_pred = []
    for path, label in test_entries:
        text = load_email_content(path)
        tokens = preprocess_text(text, use_stemming)
        shingles = get_shingles(tokens, k=shingle_k)
        m = build_minhash_from_shingles(shingles, num_perm=num_perm)
        matches = lsh.query(m)  # lista dopasowanych dokumentów treningowych
        if matches:
            # głosowanie większościowe etykiet
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
```

---

**9. Funkcja `main`**

**Wejście:**  
- Brak parametrów wejściowych

**Wyjście:**  
- Brak bezpośredniego wyjścia (funkcja wykonuje program i zapisuje wyniki do pliku)

**Opis:**  
Główna funkcja programu koordynująca cały proces klasyfikacji LSH: wczytuje i tasuje dane, dzieli na zbiory treningowe i testowe, przygotowuje MinHash dla danych treningowych, testuje różne wartości threshold dla LSH, oblicza metryki wydajności dla każdego threshold i zapisuje szczegółowe wyniki do pliku. Dla każdego threshold buduje nowy indeks LSH i przeprowadza klasyfikację.

**Kod:**  
``` python
def main():
    print("📂 Wczytywanie indexu i danych...")
    index_entries = load_index(INDEX_PATH)
    random.shuffle(index_entries)

    if SAMPLE_SIZE:
        index_entries = index_entries[:SAMPLE_SIZE]
        print(f"⚠️ SAMPLE_SIZE aktywne. Wykorzystuję {len(index_entries)} pierwszych wpisów")

    split_point = int(len(index_entries) * TRAIN_RATIO)
    train_entries = index_entries[:split_point]
    test_entries = index_entries[split_point:]

    print(f"Łącznie: {len(index_entries)} dokumentów; trening: {len(train_entries)}; test: {len(test_entries)}")
    results_lines = []
    results_lines.append(f"LSH MinHash results\nSAMPLE_SIZE={SAMPLE_SIZE}\nNUM_PERM={NUM_PERM}\nSHINGLE_SIZE={SHINGLE_SIZE}\nUSE_STEMMING={USE_STEMMING}\n")

    # Przygotowuje MinHash na treningu (raz). Będzie ono wstawiane do nowych LSH dla różnych thresholdów
    print("🧠 Budowanie MinHash dla zbioru treningowego...")
    t0 = time.time()
    train_mh_map, train_label_map = prepare_train_min_hashes(train_entries, use_stemming=USE_STEMMING,
                                                            shingle_k=SHINGLE_SIZE, num_perm=NUM_PERM)
    t_prep = time.time() - t0
    print(f"Gotowe. Czas przygotowania MinHash treningu: {t_prep:.2f}s")
    results_lines.append(f"prepare_time={t_prep:.2f}s\n")

    # Dla każdego threshold buduje nowy MinHashLSH (z tym samym num_perm) i wstawia minhashy treningowe
    for thresh in THRESHOLDS:
        print(f"\n🔎 Test dla threshold = {thresh}")
        results_lines.append(f"\nTHRESHOLD={thresh}\n")
        # buduje LSH z parametrem threshold
        t0 = time.time()
        lsh = MinHashLSH(threshold=thresh, num_perm=NUM_PERM)
        # wstawia minhashy treningowe
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
        print(f"🎯 Accuracy: {acc:.2f}% | ⏱ Czas tworzenia LSH: {build_time:.2f}s | ⏱ Czas klasyfikacji LSH: {elapsed:.2f}s")
        print("📊 Confusion matrix (%):")
        print(f"      spam      ham")
        print(f"spam  {cm_percent[0,0]:6.2f}%   {cm_percent[0,1]:6.2f}%")
        print(f"ham   {cm_percent[1,0]:6.2f}%   {cm_percent[1,1]:6.2f}%")

        # zapis wyników
        results_lines.append(f"accuracy={acc:.2f}%\n")
        results_lines.append(f"build_time={build_time:.2f}s classify_time={elapsed:.2f}s\n")
        results_lines.append("confusion_percent:\n")
        results_lines.append(f"spam_spam={cm_percent[0,0]:6.2f}% spam_ham={cm_percent[0,1]:6.2f}%\n")
        results_lines.append(f"ham_spam={cm_percent[1,0]:6.2f}% ham_ham={cm_percent[1,1]:6.2f}%\n")

    # zapis do pliku
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(results_lines))

    print(f"\n📁 Wyniki zapisano do: {RESULTS_FILE}")
```

---

**10. Kompletny kod**  
Poniżej znajduje się kompletny kod programu, który można uruchomić.

**Kod:**  
``` python
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
INDEX_PATH = "trec07p/full/index"       # ścieżka do indexu
DATA_PATH = "trec07p"                   # ścieżka do danych
TRAIN_RATIO = 0.8                       # stosunek danych treningowych do testowych
SAMPLE_SIZE = None                      # ograniczenie liczby próbek, np. 2000 dla testów, None = całość
RESULTS_FILE = "results_lsh.txt"        # nazwa pliku wynikowego

# Parametry LSH / MinHash
NUM_PERM = 128                          # liczba permutacji w MinHash
SHINGLE_SIZE = 3                        # rozmiar shingli (k-gramów)
USE_STEMMING = True                     # czy stosować stemizację
THRESHOLDS = [0.1, 0.3, 0.5, 0.7, 0.9]  # testowane progi LSH
DEFAULT_LABEL = "ham"                   # etykieta domyślna, gdy brak dopasowań w LSH

random.seed(42)                         # ustawienie ziarna losowości


# === POMOCNICZE FUNKCJE ===
# wczytuje index plików i etykiet
def load_index(index_path):
    entries = []
    with open(index_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                label, path = parts[0], parts[1]
                # Normalizuje ścieżkę: '../data/inmail.X' -> 'trec07p/data/inmail.X'
                full_path = os.path.join(DATA_PATH, path.replace("../", ""))
                entries.append((full_path, label))
    return entries

# Wczytuje zawartość e-maila i zwraca string tekstowy (ignoruje błędy kodowania)
def load_email_content(filepath):
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


# Przetwarza tekst: czyszczenie, tokenizacja, usuwanie stopwords i (opcjonalnie) stemizacja. zwraca listę tokenów
def preprocess_text(text, use_stemming=True):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    sw = set(stopwords.words("english"))
    tokens = [t for t in tokens if t.isalpha() and t not in sw]
    if use_stemming:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(t) for t in tokens]
    return tokens

# Zwraca listę shingli (k-gramów) utworzonych z tokenów (continuous k-word shingles)
def get_shingles(tokens, k=3):
    if len(tokens) < k:
        # fallback: użyj pojedynczych tokenów
        return tokens
    shingles = []
    for i in range(len(tokens) - k + 1):
        sh = " ".join(tokens[i:i + k])
        shingles.append(sh)
    return shingles


# Zwraca MinHash obliczony na zestawie shingles (unikatowych).
def build_minhash_from_shingles(shingles, num_perm=128):
    m = MinHash(num_perm=num_perm)
    # używamy zestawu, aby uniknąć wielokrotnego dodawania tego samego shingla
    for s in set(shingles):
        m.update(s.encode("utf8"))
    return m


# === PROCEDURY TRENING / TEST ===
# Przygotowuje MinHash dla każdego dokumentu treningowego i mapuje identyfikatory na etykiety. Zwraca dwa słowniki 
def prepare_train_min_hashes(train_entries, use_stemming=True, shingle_k=3, num_perm=128):
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


# Dla każdego dokumentu testowego: oblicza MinHash, pyta LSH o dopasowania, jeśli lista niepusta - dokonuje głosowania etykiet (majority vote), jeśli pusta - przypisuje DEFAULT_LABEL
def classify_with_lsh(lsh, train_label_map, test_entries, use_stemming=True, shingle_k=3, num_perm=128):
    y_true = []
    y_pred = []
    for path, label in test_entries:
        text = load_email_content(path)
        tokens = preprocess_text(text, use_stemming)
        shingles = get_shingles(tokens, k=shingle_k)
        m = build_minhash_from_shingles(shingles, num_perm=num_perm)
        matches = lsh.query(m)  # lista dopasowanych dokumentów treningowych
        if matches:
            # głosowanie większościowe etykiet
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
        print(f"⚠️ SAMPLE_SIZE aktywne. Wykorzystuję {len(index_entries)} pierwszych wpisów")

    split_point = int(len(index_entries) * TRAIN_RATIO)
    train_entries = index_entries[:split_point]
    test_entries = index_entries[split_point:]

    print(f"Łącznie: {len(index_entries)} dokumentów; trening: {len(train_entries)}; test: {len(test_entries)}")
    results_lines = []
    results_lines.append(f"LSH MinHash results\nSAMPLE_SIZE={SAMPLE_SIZE}\nNUM_PERM={NUM_PERM}\nSHINGLE_SIZE={SHINGLE_SIZE}\nUSE_STEMMING={USE_STEMMING}\n")

    # Przygotowuje MinHash na treningu (raz). Będzie ono wstawiane do nowych LSH dla różnych thresholdów
    print("🧠 Budowanie MinHash dla zbioru treningowego...")
    t0 = time.time()
    train_mh_map, train_label_map = prepare_train_min_hashes(train_entries, use_stemming=USE_STEMMING,
                                                            shingle_k=SHINGLE_SIZE, num_perm=NUM_PERM)
    t_prep = time.time() - t0
    print(f"Gotowe. Czas przygotowania MinHash treningu: {t_prep:.2f}s")
    results_lines.append(f"prepare_time={t_prep:.2f}s\n")

    # Dla każdego threshold buduje nowy MinHashLSH (z tym samym num_perm) i wstawia minhashy treningowe
    for thresh in THRESHOLDS:
        print(f"\n🔎 Test dla threshold = {thresh}")
        results_lines.append(f"\nTHRESHOLD={thresh}\n")
        # buduje LSH z parametrem threshold
        t0 = time.time()
        lsh = MinHashLSH(threshold=thresh, num_perm=NUM_PERM)
        # wstawia minhashy treningowe
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
        print(f"🎯 Accuracy: {acc:.2f}% | ⏱ Czas tworzenia LSH: {build_time:.2f}s | ⏱ Czas klasyfikacji LSH: {elapsed:.2f}s")
        print("📊 Confusion matrix (%):")
        print(f"      spam      ham")
        print(f"spam  {cm_percent[0,0]:6.2f}%   {cm_percent[0,1]:6.2f}%")
        print(f"ham   {cm_percent[1,0]:6.2f}%   {cm_percent[1,1]:6.2f}%")

        # zapis wyników
        results_lines.append(f"accuracy={acc:.2f}%\n")
        results_lines.append(f"build_time={build_time:.2f}s classify_time={elapsed:.2f}s\n")
        results_lines.append("confusion_percent:\n")
        results_lines.append(f"spam_spam={cm_percent[0,0]:6.2f}% spam_ham={cm_percent[0,1]:6.2f}%\n")
        results_lines.append(f"ham_spam={cm_percent[1,0]:6.2f}% ham_ham={cm_percent[1,1]:6.2f}%\n")

    # zapis do pliku
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(results_lines))

    print(f"\n📁 Wyniki zapisano do: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
```

#### Wyniki

``` text
📂 Wczytywanie indexu i danych...
Łącznie: 75419 dokumentów; trening: 60335; test: 15084
🧠 Budowanie MinHash dla zbioru treningowego...
Gotowe. Czas przygotowania MinHash treningu: 452.07s

🔎 Test dla threshold = 0.1
LSH zbudowano w 9.47s
🎯 Accuracy: 94.50% | ⏱ Czas tworzenia LSH: 9.47s | ⏱ Czas klasyfikacji LSH: 129.05s
📊 Confusion matrix (%):
      spam      ham
spam  60.55%    5.32%
ham   0.18%     33.94%

🔎 Test dla threshold = 0.3
LSH zbudowano w 6.92s
🎯 Accuracy: 88.80% | ⏱ Czas tworzenia LSH: 6.92s | ⏱ Czas klasyfikacji LSH: 120.82s
📊 Confusion matrix (%):
      spam      ham
spam  54.76%    11.12%
ham   0.09%     34.04%

🔎 Test dla threshold = 0.5
LSH zbudowano w 4.78s
🎯 Accuracy: 79.14% | ⏱ Czas tworzenia LSH: 4.78s | ⏱ Czas klasyfikacji LSH: 118.51s
📊 Confusion matrix (%):
      spam      ham
spam  45.09%    20.79%
ham   0.07%     34.06%

🔎 Test dla threshold = 0.7
LSH zbudowano w 3.11s
🎯 Accuracy: 70.72% | ⏱ Czas tworzenia LSH: 3.11s | ⏱ Czas klasyfikacji LSH: 116.56s
📊 Confusion matrix (%):
      spam      ham
spam  36.65%    29.23%
ham   0.05%     34.07%

🔎 Test dla threshold = 0.9
LSH zbudowano w 1.53s
🎯 Accuracy: 62.70% | ⏱ Czas tworzenia LSH: 1.53s | ⏱ Czas klasyfikacji LSH: 115.94s
📊 Confusion matrix (%):
      spam      ham
spam  28.61%    37.26%
ham   0.04%     34.08%

📁 Wyniki zapisano do: results_lsh.txt
```

### Zadanie 5
Dokonać klasyfikacji binarnej wiadomości z archiwum (zadanie 1) na spam i ham, stosując algorytm Naive Bayes.

**Uwagi:**
1. Do realizacji zadania należy użyć implementacji algorytmu z biblioteki Scikit-learn. Algorytm dostępny jest poprzez obiekt MultinomialNB.
2. Porównać działanie algorytmu dla przypadków:
   - algorytm pracuje na całych tematach i ciele wiadomości w postaci zwykłego tekstu bez usuwania słów przestankowych i stemizacji przy pomocy narzędzi z biblioteki NLTK.
   - algorytm pracuje na bazie stemizowanych danych z usuniętymi słowami przestankowymi.
1. Uzyskane wyniki przedstawić przy pomocy macierzy konfuzji i wskaźnika accuracy.
2. Porównać uzyskane wyniki do wyników uzyskanych przy zastosowaniu metod z poprzednich zadań.

#### Implementacja

**1. Konfiguracja globalna**

Na wstępie programu znajduje się kod, który definiuje stałe konfiguracyjne używane w całym programie. Ułatwia to dostosowanie parametrów bez konieczności modyfikowania logiki programu.

**Kod:**  
``` python
INDEX_PATH = "trec07p/full/index"       # ścieżka do indexu
DATA_PATH = "trec07p"                   # ścieżka do danych
TRAIN_RATIO = 0.8                       # stosunek danych treningowych do testowych
SAMPLE_SIZE = None                      # ograniczenie liczby próbek, np. 2000 dla testów, None = całość
RESULTS_FILE = "results_naive_bayes.txt"# nazwa pliku wynikowego

random.seed(42)                         # ustawienie ziarna losowości
```

**2. Funkcja `load_index`**

**Wejście:**  
- `index_path` (string) - ścieżka do pliku z indeksem wiadomości

**Wyjście:**  
- `entries` (list) - lista krotek zawierających pełną ścieżkę do pliku i etykietę (spam/ham)

**Opis:**  
Funkcja wczytuje plik indeksu TREC07P, parsuje każdą linię rozdzielając ją na etykietę (spam/ham) i ścieżkę do pliku. Tworzy pełne ścieżki do plików przez połączenie ścieżki bazowej DATA_PATH ze ścieżką z indeksu (po usunięciu "../"). Zwraca listę wszystkich wpisów gotowych do przetwarzania.

**Kod:**  
``` python
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
```

---

**3. Funkcja `load_email_content`**

**Wejście:**  
- `filepath` (string) - ścieżka do pliku z wiadomością email

**Wyjście:**  
- `text` (string) - połączony temat i treść wiadomości lub pusty string w przypadku błędu

**Opis:**  
Funkcja wczytuje i parsuje wiadomość email, wyciągając zarówno temat (Subject) jak i treść wiadomości. Obsługuje wiadomości wieloczęściowe (multipart) - iteruje przez wszystkie części i wyciąga tylko te o typie tekstowym. Łączy temat z treścią w jeden string, co zapewnia, że algorytm Naive Bayes będzie wykorzystywał całą dostępną informację tekstową.

**Kod:**  
``` python
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
```

---

**4. Funkcja `preprocess_text`**

**Wejście:**  
- `text` (string) - tekst wiadomości email do przetworzenia

**Wyjście:**  
- `text` (string) - przetworzony tekst po stemizacji i usunięciu stopwords

**Opis:**  
Funkcja przeprowadza pełne przetwarzanie tekstu NLTK: konwersja na małe litery, usuwanie znaków interpunkcyjnych, tokenizacja na pojedyncze słowa, filtrowanie tylko słów alfabetycznych, usuwanie stopwords (słów bez znaczenia) oraz stemizacja przy użyciu algorytmu PorterStemmer. Na końcu łączy tokeny z powrotem w string dla kompatybilności z CountVectorizer.

**Kod:**  
``` python
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    sw = set(stopwords.words("english"))
    tokens = [t for t in tokens if t.isalpha() and t not in sw]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(t) for t in tokens]
    return " ".join(tokens)
```

---

**5. Funkcja `prepare_data`**

**Wejście:**  
- `entries` (list) - lista krotek (ścieżka, etykieta) do przetworzenia
- `use_preprocessing` (bool) - flaga określająca czy stosować przetwarzanie NLTK

**Wyjście:**  
- `texts` (list) - lista tekstów wiadomości (przetworzonych lub nie)
- `labels` (list) - lista etykiet (spam/ham)

**Opis:**  
Funkcja przetwarza wszystkie dokumenty z podanej listy. Dla każdego dokumentu wczytuje treść emaila i opcjonalnie stosuje preprocessing NLTK w zależności od parametru `use_preprocessing`. Zwraca dwie listy: tekstów przygotowanych do wektoryzacji oraz odpowiadających im etykiet.

**Kod:**  
``` python
def prepare_data(entries, use_preprocessing=False):
    texts, labels = [], []
    for path, label in entries:
        text = load_email_content(path)
        if use_preprocessing:
            text = preprocess_text(text)
        texts.append(text)
        labels.append(label)
    return texts, labels
```

---

**6. Funkcja `run_naive_bayes`**

**Wejście:**  
- `train_entries` (list) - lista krotek (ścieżka, etykieta) dla danych treningowych
- `test_entries` (list) - lista krotek (ścieżka, etykieta) dla danych testowych
- `use_preprocessing` (bool) - flaga określająca czy stosować przetwarzanie NLTK

**Wyjście:**  
- `acc` (float) - dokładność klasyfikacji w procentach
- `cm_percent` (numpy.ndarray) - macierz konfuzji w procentach
- `elapsed` (float) - czas wykonania w sekundach

**Opis:**  
Funkcja przeprowadza pełny eksperyment z klasyfikatorem Naive Bayes: przygotowuje dane treningowe i testowe, tworzy macierz cech przy użyciu CountVectorizer (bag-of-words), trenuje model MultinomialNB, dokonuje predykcji na danych testowych i oblicza metryki wydajności. Wyświetla szczegółowe wyniki w konsoli i zwraca wartości do dalszej analizy.

**Kod:**  
``` python
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
```

---

**7. Funkcja `main`**

**Wejście:**  
- Brak parametrów wejściowych

**Wyjście:**  
- Brak bezpośredniego wyjścia (funkcja wykonuje program i zapisuje wyniki do pliku)

**Opis:**  
Główna funkcja programu koordynująca eksperymenty z Naive Bayes: wczytuje i tasuje dane, dzieli na zbiory treningowe i testowe, przeprowadza dwa eksperymenty (bez przetwarzania tekstu i z pełnym przetwarzaniem NLTK), porównuje wyniki pod względem accuracy i macierzy konfuzji, oraz zapisuje szczegółowe wyniki do pliku tekstowego. Eksperymenty pozwalają na porównanie wpływu preprocessingu na skuteczność klasyfikacji.

**Kod:**  
``` python
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
```

---

**8. Kompletny kod**  
Poniżej znajduje się kompletny kod programu, który można uruchomić.

**Kod:**  
``` python
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
```


#### Wyniki

``` text
📂 Wczytywanie danych...
Łącznie: 75419 dokumentów; trening: 60335; test: 15084

🧠 Uruchamianie Naive Bayes (Bez preprocessingu)...
🎯 Accuracy: 99.24% | ⏱ Czas wykonania: 69.66s
📊 Confusion matrix (%):
      spam      ham
spam  65.46%    0.42%
ham   0.34%     33.78%

🧠 Uruchamianie Naive Bayes (z preprocessingiem)...
🎯 Accuracy: 98.72% | ⏱ Czas wykonania: 381.12s
📊 Confusion matrix (%):
      spam      ham
spam  64.85%    1.03%
ham   0.25%     33.87%

📁 Wyniki zapisano do: results_naive_bayes.txt
```

