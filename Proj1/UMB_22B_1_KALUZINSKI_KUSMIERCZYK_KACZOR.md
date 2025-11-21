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

---

**Parametry sprzętowe**:  
Eksperymenty przeprowadzono na laptopie z następującymi parametrami:
- Procesor: AMD Ryzen 5 4500U 2.38 GHz, 6 rdzeni, 6 wątków
- GPU: AMD Radeon Graphics 497 MB
- Pamięć RAM: 16 GB 2666 MHz
- System operacyjny: Windows 11 Pro 64-bit

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

#### Wnioski

W zadaniach 2 i 3 zastosowano metodę blacklisty słów kluczowych do klasyfikacji wiadomości email na spam i ham. Poniżej przedstawiono szczegóły dotyczące tej metody oraz analizę uzyskanych wyników.

**Opis metody**  
Metoda blacklisty słów kluczowych polega na identyfikacji najbardziej charakterystycznych słów dla kategorii spamu i wykorzystaniu ich do klasyfikacji nowych wiadomości. Algorytm działa w dwóch etapach:
1. **Faza treningowa**: Analiza danych treningowych w celu zidentyfikowania słów o najwyższym stosunku występowania w wiadomościach spam do ham, tworząc listę zakazanych słów kluczowych (blacklistę).
2. **Faza klasyfikacji**: Oznaczenie wiadomości jako spam, jeśli ta zawiera którekolwiek ze słów z blacklisty; w przeciwnym razie oznaczenie jako ham.

**Zalety metody**  
- **Prostota implementacji** - niewielka złożoność obliczeniowa
- **Łatwość interpretacji** - możliwość analizy które słowa decydują o klasyfikacji
- **Szybkość klasyfikacji** - w fazie predykcji wymaga tylko sprawdzenia obecności słów
- **Niskie wymagania pamięciowe** - przechowuje tylko listę słów kluczowych

**Wady metody**  
- **Niska dokładność** - prosty model może nie uchwycić złożonych wzorców w danych
- **Podatność na zmiany** - nowe formy spamu mogą ominąć istniejącą blacklistę
- **Brak kontekstu** - nie uwzględnia relacji między słowami ani ich kolejności
- **Problemy z fałszywymi pozytywami** - słowa mogą mieć różne znaczenia w różnych kontekstach

---

Program realizował dwa testy: z zastosowaniem stemizacji oraz bez niej.  
**Stemizacja** to proces redukcji słów do ich formy podstawowej, poprzez usunięcie prefiksów i sufiksów. Na przykład:
- "running", "runs", "ran" → "run"
- "connection", "connected", "connecting" → "connect"

Celem stosowania stemizacji jest zredukowanie wymiarowości danych tekstowych poprzez grupowanie różnych form tego samego słowa, co powinno poprawić skuteczność klasyfikacji poprzez lepsze uogólnienie wzorców.

---

**Konfiguracja programu**  
Na wstępie określono parametry eksperymentu, takie jak ścieżki do danych, stosunek podziału na zbiór treningowy i testowy oraz liczba słów w blacklist. Najważniejsze parametry to:
- **`TRAIN_RATIO = 0.8`** - Standardowy podział 80/20, który jest powszechnie stosowany w uczeniu maszynowym, zapewniając wystarczającą ilość danych do treningu (60,335 wiadomości) przy zachowaniu reprezentatywnego zbioru testowego (15,084 wiadomości).
- **`TOP_N = 100`** - Limit 100 słów w blackliście stanowi kompromis między skutecznością a specyficznością. Zbyt mała lista mogłaby pomijać istotne wzorce, a zbyt duża zwiększałaby ryzyko nadmiernego dopasowania.
- **`SAMPLE_SIZE = None`** - Użycie całego zbioru danych zapewnia wiarygodność wyników, jednak parametr umożliwia szybkie testy na mniejszych próbkach podczas rozwoju algorytmu.

---

**Analiza wyników**  
Poniższa tabela przedstawia porównanie kluczowych metryk uzyskanych w obu testach:

| Metryka | Ze stemizacją | Bez stemizacji | Różnica | Wnioski |
|---------|---------------|----------------|---------|---------|
| **Accuracy** | 61.83% | 58.64% | **+3.20%** | Stemizacja poprawia ogólną dokładność klasyfikacji |
| **Czas wykonania** | 2465.29s (≈41 min) | 239.89s (≈4 min) | **+2225.39s** | Stemizacja znacząco wydłuża czas przetwarzania |
| **Poprawny spam** | 28.65% | 25.45% | **+3.20%** | Lepsze wykrywanie wiadomości spam |
| **Fałszywe negatywy** | 38.07% | 41.28% | **-3.21%** | Mniej spamu przechodzi niezauważone |
| **Fałszywe pozytywy** | 0.09% | 0.09% | **0.00%** | Brak wpływu na błędne oznaczanie ham |
| **Poprawny ham** | 33.18% | 33.19% | **-0.01%** | Klasyfikacja ham pozostaje niezmienna |

**Efektywność stemizacji**  
Stemizacja przynosi wymierne korzyści w skuteczności klasyfikacji, zwiększając accuracy o 3.20%. Poprawa koncentruje się głównie na lepszym wykrywaniu spamu, gdzie obserwujemy wzrost poprawnie sklasyfikowanych wiadomości spam o 3.20% i redukcję fałszywych negatywów o 3.21%.

**Koszt wydajnościowy**  
Czas przetwarzania ze stemizacją jest około 10x dłuższy (2465s vs 240s), co stanowi istotny kompromis w zastosowaniach wymagających szybkiego przetwarzania dużych zbiorów danych.

**Wpływ na różne kategorie wiadomości**  
- **Spam**: Stemizacja znacząco poprawia wykrywalność (+3.20%)
- **Ham**: Brak zauważalnego wpływu na klasyfikację
- **Fałszywe pozytywy**: Minimalne i identyczne w obu wersjach (0.09%)

---

Pomimo poprawy dzięki stemizacji, ogólna dokładność na poziomie ~60% potwierdza, że prosta **blacklista** słów kluczowych ma fundamentalne ograniczenia i powinna być traktowana jako element szerszego systemu filtrowania spamu, a nie samodzielne rozwiązanie.  
Wersja ze stemizacją jest preferowana ze względu na lepsze wykrywanie spamu, jednak kosztem znacznie dłuższego czasu przetwarzania. Wybór między wersjami powinien uwzględniać specyficzne wymagania dotyczące dokładności i wydajności w danym zastosowaniu.  
Obie wersje mogą skutecznie służyć jako pierwsza linia obrony, jednak wysoki odsetek fałszywych negatywów (~38-41%) wskazuje na potrzebę dodatkowych metod weryfikacji.

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

#### Wnioski

W zadaniu 4 zastosowano algorytm Locality Sensitive Hashing (LSH) z MinHash do klasyfikacji binarnej wiadomości email na spam i ham. Przetestowano różne wartości progu (threshold) dla LSH, co miało wpływ na dokładność klasyfikacji.

**Opis algorytmu**  
Algorytm LSH (Locality-Sensitive Hashing) z wykorzystaniem MinHash to zaawansowana technika oparta na teorii prawdopodobieństwa, służąca do znajdowania podobnych dokumentów w dużych zbiorach danych. Algorytm działa w następujących etapach:
1. **Tworzenie shingli**: Podział tekstu na ciągłe sekwencje słów (k-gramy)
2. **MinHash**: Generowanie sygnatur dokumentów poprzez wielokrotne haszowanie shingli i wybieranie minimalnych wartości hash
3. **LSH**: Grupowanie podobnych dokumentów w "koszykach" na podstawie podobieństwa ich sygnatur
4. **Klasyfikacja**: Głosowanie większościowe etykiet spośród najbliższych sąsiadów w zbiorze treningowym

**Zalety metody**
- **Skalowalność** - efektywne przetwarzanie dużych zbiorów danych 
- **Odporność na permutacje** - niezależność od kolejności słów w dokumencie
- **Wykrywanie podobieństw** - zdolność do identyfikacji dokumentów o podobnej treści 
- **Probabilistyczna dokładność** - kontrola precyzji poprzez parametr threshold

**Wady metody**
- **Złożoność konfiguracji** - wymaga dostrojenia wielu parametrów (num_perm, shingle_size, threshold) w celu uzyskania optymalnych wyników
- **Koszt pamięciowy** - przechowywanie sygnatur MinHash dla wszystkich dokumentów 
- **Zależność od jakości danych** - wrażliwość na preprocessing i dobór shingli

---

**Konfiguracja programu**  
Podobnie jak poprzednio, na wstępie należało zdefiniować stałe konfiguracyjne, takie jak ścieżki do danych, parametry LSH/MinHash oraz ustawienia dotyczące przetwarzania tekstu (stemizacja, rozmiar shingli itp.). Działanie algorytmu zostało przetestowane dla różnych wartości progu (threshold) LSH: 0.1, 0.3, 0.5, 0.7, 0.9. Najważniejsze parametry to: 
- **`TRAIN_RATIO = 0.8`** - Standardowy podział 80/20 zapewnia odpowiednią ilość danych treningowych (60,335 wiadomości) przy zachowaniu reprezentatywnego zbioru testowego (15,084 wiadomości).
- **`NUM_PERM = 128`** - Liczba permutacji stanowi kompromis między dokładnością a wydajnością. Większa liczba zwiększa precyzję, ale kosztem czasu przetwarzania.
- **`SHINGLE_SIZE = 3`** - Rozmiar shingli (3-gramów) pozwala na uchwycenie kontekstu słów, co jest kluczowe dla identyfikacji podobieństw między dokumentami.
- **`USE_STEMMING = True`** - Włączenie stemizacji, co wynika z pozytywnych doświadczeń z Zadania 3, gdzie stemizacja poprawiła skuteczność klasyfikacji.
- **`THRESHOLDS = [0.1, 0.3, 0.5, 0.7, 0.9]`** - Zakres progów testowych od bardzo  niskiego (0.1) do wysokiego (0.9), pozwala na kompleksową analizę kompromisu między czułością a specyficznością.

---

**Tabela wyników programu dla różnych wartości threshold**
| Threshold | Accuracy | Czas budowy LSH | Czas klasyfikacji | Poprawny spam | Fałszywe negatywy | Fałszywe pozytywy | Poprawny ham |
|-----------|----------|-----------------|-------------------|---------------|-------------------|-------------------|-------------|
| **0.1** | **94.50%** | 9.47s | 129.05s | **60.55%** | **5.32%** | 0.18% | 33.94% |
| **0.3** | 88.80% | 6.92s | 120.82s | 54.76% | 11.12% | **0.09%** | 34.04% |
| **0.5** | 79.14% | 4.78s | 118.51s | 45.09% | 20.79% | 0.07% | 34.06% |
| **0.7** | 70.72% | 3.11s | 116.56s | 36.65% | 29.23% | 0.05% | 34.07% |
| **0.9** | 62.70% | **1.53s** | **115.94s** | 28.61% | 37.26% | **0.04%** | **34.08%** |

**Optymalizacja parametru threshold**
Analiza wyników pokazuje, że **Threshold = 0.1** osiąga najlepszą dokładność (94.50%), co wskazuje na optymalny kompromis między czułością a specyficznością. Niższe wartości threshold zwiększają liczbę dopasowań, poprawiając wykrywanie spamu kosztem niewielkiego wzrostu fałszywych pozytywów.

**Wydajność czasowa**
- **Czas przygotowania MinHash**: 452.07s - jednorazowy koszt inicjalizacji
- **Czas budowy LSH**: Maleje liniowo z wzrostem threshold (9.47s → 1.53s)
- **Czas klasyfikacji**: Stabilny na poziomie ~115-129s, niezależnie od threshold

---

**Porównanie z metodą blacklisty (Zadania 2-3)**
| Metryka | LSH (threshold=0.1) | Blacklista (ze stemizacją) | Poprawa |
|---------|---------------------|----------------------------|---------|
| **Accuracy** | **94.50%** | 61.83% | **+32.67%** |
| **Poprawny spam** | **60.55%** | 28.65% | **+31.90%** |
| **Fałszywe negatywy** | **5.32%** | 38.07% | **-32.75%** |
| **Fałszywe pozytywne** | 0.18% | **0.09%** | +0.09% |
| **Czas przetwarzania** | ~591s | 2465s | **-1874s** |

Ocena efektywności algorytmu LSH w porównaniu z metodą blacklisty wykazała jego znaczną przewagę, wyrażającą się wzrostem dokładności o 32,67% przy jednoczesnym skróceniu czasu przetwarzania.

Kluczowym czynnikiem wpływającym na skuteczność metody jest odpowiedni dobór parametru threshold, od którego zależy kompromis między czułością a specyficznością klasyfikatora. Ponadto algorytm LSH wyróżnia się doskonałą skalowalnością, zapewniając przewidywalne czasy przetwarzania nawet przy pracy na dużych zbiorach danych. Warto podkreślić, że we wszystkich testowanych konfiguracjach utrzymał on bardzo niski poziom fałszywych trafień, gdzie błędne oznaczanie prawidłowych wiadomości jako spam nie przekroczyło 0,2%.

W kontekście praktycznych zastosowań, dla systemów produkcyjnych rekomendowane jest ustawienie threshold na poziomie 0,1, co gwarantuje wysoką skuteczność wykrywania spamu przy zachowaniu akceptowalnego odsetka fałszywych alarmów.

Metoda LSH z wykorzystaniem MinHash okazała się zdecydowanie bardziej efektywna niż prosta blacklista słów kluczowych, stanowiąc profesjonalne i gotowe do wdrożenia rozwiązanie do klasyfikacji wiadomości email na skalę przemysłową.

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

#### Wnioski

W zadaniu 5 zastosowano klasyfikator Naive Bayes (MultinomialNB) do binarnej klasyfikacji wiadomości email na spam i ham. Przeprowadzono dwa eksperymenty: jeden bez preprocessingu tekstu, a drugi z pełnym przetwarzaniem NLTK (usuwanie stopwords i stemizacja).

**Opis algorytmu**  
Klasyfikator Naive Bayes to probabilistyczny algorytm oparty na twierdzeniu Bayesa, który zakłada niezależność cech (słów) przy danej etykiecie. Algorytm działa w następujących etapach:
1. **Ekstrakcja cech**: Przekształcenie tekstu na reprezentację numeryczną (Bag-of-Words)
2. **Trening**: Obliczenie prawdopodobieństw warunkowych dla każdego słowa w kontekście klas spam/ham
3. **Klasyfikacja**: Obliczenie prawdopodobieństwa posterior dla nowej wiadomości i przypisanie do klasy o wyższym prawdopodobieństwie

**Zalety metody**
- **Wysoka skuteczność** - doskonałe wyniki w klasyfikacji tekstu
- **Szybkość treningu** - efektywne obliczenia probabilistyczne
- **Skalowalność** - dobre działanie na dużych zbiorach danych
- **Odporność na szum** - stabilność wobec częściowo nieistotnych cech

**Wady metody**
- **Założenie niezależności** - nierealistyczne założenie o niezależności słów
- **Wrażliwość na rzadkie słowa** - problem z wyrazami nieobecnymi w zbiorze treningowym
- **Zależność od preprocessingu** - wyniki mogą się różnić w zależności od przygotowania danych

---

**Konfiguracja programu**  
Podobnie jak poprzednio, na wstępie należało zdefiniować stałe konfiguracyjne. Konfiguracja programu została zaprojektowana w celu porównania wpływu preprocessingu tekstu na skuteczność algorytmu, dlatego eksperyment obejmuje dwa scenariusze: pracę na surowym tekście oraz na danych po pełnym przetworzeniu NLTK. Najważniejsze parametry to:
- **`TRAIN_RATIO = 0.8`** - Standardowy podział 80/20, który jest powszechnie stosowany w uczeniu maszynowym, zapewniając wystarczającą ilość danych do treningu (60,335 wiadomości) przy zachowaniu reprezentatywnego zbioru testowego (15,084 wiadomości).
- **`SAMPLE_SIZE = None`** - Użycie całego zbioru danych zapewnia wiarygodność wyników, jednak parametr umożliwia szybkie testy na mniejszych próbkach podczas rozwoju algorytmu.
- **`random.seed(42)`** - Gwarancja powtarzalności eksperymentów poprzez ustalenie ziarna losowości.

---

**Tabela porównawcza wyników:**
| Metryka | Bez preprocessingu | Z preprocessingiem | Różnica | Wnioski |
|---------|-------------------|-------------------|---------|---------|
| **Accuracy** | **99.24%** | 98.72% | **-0.52%** | Preprocessing nieznacznie obniża dokładność |
| **Czas wykonania** | **69.66s** | 381.12s | **+311.46s** | Preprocessing znacząco wydłuża czas |
| **Poprawny spam** | **65.46%** | 64.85% | **-0.61%** | Nieznacznie lepsze bez preprocessingu |
| **Fałszywe negatywy** | **0.42%** | 1.03% | **+0.61%** | Więcej spamu przechodzi z preprocessingiem |
| **Fałszywe pozytywne** | 0.34% | **0.25%** | **-0.09%** | Preprocessing redukuje błędy ham→spam |
| **Poprawny ham** | 33.78% | **33.87%** | **+0.09%** | Nieznacznie lepsze z preprocessingiem |

**Wersja bez preprocessingu osiąga nieznacznie lepszą dokładność (99.24% vs 98.72%)**, co jest niespodziewanym wynikiem, ponieważ preprocessing teoretycznie powinien poprawiać jakość cech. Sugeruje to, że niektóre słowa pomocnicze mogą być charakterystyczne dla spamu i ich usunięcie obniża skuteczność klasyfikacji. Dodatkowo można założyć, że znaki specjalne mogą być istotnymi wskaźnikami spamu (np. `!!!`, `$$$$`), a ich usunięcie w preprocessingie prowadzi do utraty informacji. Co więcej, sprowadzenie słów do ich podstawowych form (stemizacja) może usuwać subtelne różnice między wyrazami, które są istotne dla klasyfikacji. 

**Preprocessing zwiększa czas wykonania ponad 5-krotnie** (69.66s → 381.12s), co wynika z dodatkowych operacji lingwistycznych na każdym dokumencie.

W rezultacie możemy przyjąć poniższe podejście w zależności od priorytetów systemu:
- **Bez preprocessingu**: Lepsze wykrywanie spamu, ale więcej fałszywych pozytywnych i szybsze działanie
- **Z preprocessingiem**: Gorsze wykrywanie spamu, ale mniej fałszywych pozytywnych kosztem czasu

---

**Porównanie z metodami z poprzednich zadań**
| Metoda | Najlepsza accuracy | Czas przetwarzania | Zalety | Wady |
|--------|-------------------|-------------------|--------|------|
| **Blacklista** | 61.83% | 2465s | Prosta, interpretowalna | Niska skuteczność |
| **LSH (threshold=0.1)** | 94.50% | 591s | Skalowalna, dobre podobieństwa | Zależna od parametrów |
| **Naive Bayes** | **99.24%** | **70s** | **Najwyższa dokładność**, szybki | Założenie niezależności |

Metoda Naive Bayes wyraźnie dominuje nad pozostałymi testowanymi rozwiązaniami, osiągając najwyższą skuteczność klasyfikacji na poziomie 99,24%. Wynik ten znacząco przewyższa zarówno efektywność metody opartej na LSH, jak i klasycznej blacklisty – odpowiednio o 4,74% oraz aż o 37,41%. Co ważne, analiza pokazała, że stosowanie preprocessingu nie zawsze przekłada się na poprawę jakości klasyfikacji. W przypadku Naive Bayes prostsze podejście, pozbawione dodatkowego czyszczenia danych, okazało się nie tylko skuteczniejsze, ale także szybsze. Niewielki spadek dokładności przy użyciu preprocessingu może wynikać z utraty pewnych istotnych informacji, takich jak stopwords czy interpunkcja, które – choć często traktowane jako szum – mogą w niektórych przypadkach nieść wartościowe wskazówki dla klasyfikatora.

W praktyce sugeruje się jednak korzystanie z wersji metody Naive Bayes bez preprocessingu, szczególnie w środowiskach produkcyjnych, gdzie kluczowe są zarówno wysoka dokładność, jak i krótki czas przetwarzania. Wyniki badań potwierdzają znaczną przewagę tego algorytmu nad wcześniej stosowanymi metodami, co czyni go optymalnym wyborem w zadaniach związanych z filtrowaniem spamu i innymi formami klasyfikacji tekstu.

### Zadanie 6
Dokonać klasyfikacji binarnej wiadomości z archiwum (zadanie 1) na spam i ham, stosując model gęsto łączonej głębokiej sieci neuronowej i technikę uczenia nadzorowanego.
**Uwagi:**
1. Zaproponować sposób translacji danych wejściowych do postaci akceptowanego przez sieć tensora wejściowego.
2. Zaproponować liczbę warstw ukrytych oraz liczbę węzłów w poszczególnych warstwach.
3. Zaproponować funkcje aktywacji dla węzłów w warstwach ukrytych oraz w warstwie wyjściowej.
4. Zaproponować metrykę dokładności.
5. Zaproponować optymalizator.
6. Do realizacji zadania zastosować narzędzia z biblioteki TensorFLow.
7. W wyniku realizacji zadania wygenerować macierz konfuzji oraz wartość wskaźnika accuracy.
8. Porównać uzyskane wyniki dla różnych modeli (to znaczy: ilości warstw ukrytych, ilości węzłów w warstwach, funkcji aktywacji).
9. Porównać uzyskane wyniki z wynikami uzyskanym w ramach realizacji poprzednich zadań.

#### Implementacja

**1. Konfiguracja globalna**
Na wstępie programu znajduje się kod, który definiuje stałe konfiguracyjne używane w całym programie. Ułatwia to dostosowanie parametrów bez konieczności modyfikowania logiki programu.

**Kod:**  
``` python
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
```

---

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
Funkcja wczytuje i parsuje wiadomość email, wyciągając zarówno temat (Subject) jak i treść wiadomości. Obsługuje wiadomości wieloczęściowe (multipart) - iteruje przez wszystkie części i wyciąga tylko te o typie tekstowym. Łączy temat z treścią w jeden string. Dekoduje zawartość binarną i obsługuje błędy kodowania przy użyciu kodowania latin-1.

**Kod:**  
``` python
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
```

---

**4. Funkcja `preprocess_text`**

**Wejście:**  
- `text` (string) - tekst wiadomości email do przetworzenia

**Wyjście:**  
- `text` (string) - przetworzony tekst po stemizacji i usunięciu stopwords

**Opis:**  
Funkcja przeprowadza pełne przetwarzanie tekstu NLTK: konwersja na małe litery, usuwanie znaków interpunkcyjnych, tokenizacja na pojedyncze słowa, filtrowanie tylko słów alfabetycznych, usuwanie stopwords (słów bez znaczenia) oraz stemizacja przy użyciu algorytmu PorterStemmer. Na końcu łączy tokeny z powrotem w string dla kompatybilności z TF-IDF Vectorizer.

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

**5. Funkcja `prepare_corpus`**

**Wejście:**  
- `entries` (list) - lista krotek (ścieżka, etykieta) do przetworzenia
- `use_preprocessing` (bool) - flaga określająca czy stosować przetwarzanie NLTK
- `sample_size` (int) - ograniczenie liczby przetwarzanych dokumentów

**Wyjście:**  
- `texts` (list) - lista tekstów wiadomości (przetworzonych lub nie)
- `labels` (numpy.ndarray) - tablica etykiet numerycznych (spam=1, ham=0)

**Opis:**  
Funkcja przetwarza wszystkie dokumenty z podanej listy. Dla każdego dokumentu wczytuje treść emaila i opcjonalnie stosuje preprocessing NLTK. Konwertuje etykiety tekstowe na numeryczne (spam=1, ham=0) dla kompatybilności z TensorFlow. Zwraca listę tekstów i tablicę etykiet numerycznych.

**Kod:**  
``` python
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
```

---

**6. Funkcja `build_vectorizer`**

**Wejście:**  
- `texts` (list) - lista tekstów do wektoryzacji
- `max_features` (int) - maksymalna liczba cech w wektorze TF-IDF

**Wyjście:**  
- `vec` (TfidfVectorizer) - wytrenowany obiekt vectorizera
- `X` (scipy.sparse matrix) - macierz cech w formacie TF-IDF

**Opis:**  
Funkcja tworzy i trenuje vectorizer TF-IDF na podanych tekstach. Używa zakresu n-gramów (1,2), co oznacza, że uwzględnia zarówno pojedyncze słowa jak i pary kolejnych słów. Ogranicza liczbę cech do `max_features` w celu kontroli wymiarowości danych. Zwraca wytrenowany vectorizer i przekształconą macierz cech.

**Kod:**  
``` python
def build_vectorizer(texts, max_features=20000):
    vec = TfidfVectorizer(max_features=max_features, ngram_range=(1,2))
    X = vec.fit_transform(texts)
    return vec, X
```

---

**7. Funkcja `build_model`**

**Wejście:**  
- `input_dim` (int) - wymiarowość danych wejściowych
- `layer_sizes` (list) - lista określająca liczbę neuronów w kolejnych warstwach
- `activation_hidden` (string) - funkcja aktywacji dla warstw ukrytych
- `dropout` (float) - współczynnik dropout dla regularyzacji
- `lr` (float) - learning rate dla optymalizatora

**Wyjście:**  
- `model` (Sequential) - skompilowany model sieci neuronowej

**Opis:**  
Funkcja buduje sekwencyjny model DNN zgodnie z podaną architekturą. Tworzy warstwy gęste z określoną liczbą neuronów i funkcjami aktywacji. Po każdej warstwie dodaje warstwę Dropout dla zapobiegania przeuczeniu. Ostatnia warstwa używa funkcji sigmoid dla klasyfikacji binarnej. Kompiluje model z optymalizatorem Adam, funkcją straty binary_crossentropy i metryką accuracy.

**Kod:**  
``` python
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
```

---

**8. Funkcja `main`**

**Wejście:**  
- Brak parametrów wejściowych

**Wyjście:**  
- Brak bezpośredniego wyjścia (funkcja wykonuje program i zapisuje wyniki do pliku)

**Opis:**  
Główna funkcja programu koordynująca cały proces: wczytuje i tasuje dane, przygotowuje korpus tekstowy, tworzy wektory TF-IDF, testuje różne konfiguracje modeli DNN, trenuje modele, dokonuje predykcji, oblicza metryki wydajności i zapisuje szczegółowe wyniki do pliku. Dla każdej konfiguracji modelu z listy MODEL_CONFIGS przeprowadza pełny cykl treningu i ewaluacji.

**Kod:**  
``` python
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
```


#### Wyniki

Z wyników zostały usunięte komunikaty oraz warningi TensorFlow dla zwiększenia czytelności wyników.

``` text
📂 Wczytywanie danych...
🧾 Przygotowanie korpusu tekstów (preprocessing = True)...
Przygotowano 75419 dokumentów.
Trening: 60335, Test: 15084
🔤 Tworzenie TF-IDF (max_features=20000)...
Konwersja do macierzy gęstych...
Input dim = 20000

=== Model: small | layers=[64] | activation=relu ===
Trening zakończony w 34.64s
🎯 Accuracy: 99.67% | Czas treningu: 34.64s | Czas predykcji: 1.32s
📊 Confusion matrix (%):
      spam      ham
spam  65.78%    0.09%
ham   0.24%     33.88%

=== Model: medium | layers=[128, 64] | activation=relu ===
Trening zakończony w 56.67s
🎯 Accuracy: 99.64% | Czas treningu: 56.67s | Czas predykcji: 1.03s
📊 Confusion matrix (%):
      spam      ham
spam  65.78%    0.09%
ham   0.27%     33.85%

=== Model: large | layers=[256, 128, 64] | activation=relu ===
Trening zakończony w 99.26s
🎯 Accuracy: 99.61% | Czas treningu: 99.26s | Czas predykcji: 1.17s
📊 Confusion matrix (%):
      spam      ham
spam  65.76%    0.12%
ham   0.27%     33.85%

=== Model: small_tanh | layers=[64] | activation=tanh ===
Trening zakończony w 31.32s
🎯 Accuracy: 99.64% | Czas treningu: 31.32s | Czas predykcji: 0.83s
📊 Confusion matrix (%):
      spam      ham
spam  65.77%    0.11%
ham   0.25%     33.87%

📁 Wyniki zapisano do: results_dnn.txt
```

#### Wnioski

W zadaniu 6 zastosowano gęsto łączoną głęboką sieć neuronową (DNN) do klasyfikacji binarnej wiadomości email na spam i ham. Wykorzystano różne konfiguracje architektury sieci, testując modele o różnej liczbie warstw ukrytych i liczbie neuronów w każdej warstwie.

**Opis metody**  
Metoda wykorzystuje gęsto łączoną sieć neuronową (DNN) do klasyfikacji wiadomości email, połączoną z techniką ekstrakcji cech TF-IDF. Algorytm działa w następujących etapach:
1. **Preprocessing tekstu**: Stemizacja i usuwanie stopwords przy użyciu NLTK 
2. **Ekstrakcja cech TF-IDF**: Przekształcenie tekstu na wektory numeryczne z użyciem n-gramów (1,2)
3. **Architektura DNN**: Wielowarstwowa sieć neuronowa z warstwami gęstymi i dropout dla regularyzacji:
   - Warstwy ukryte: Różne konfiguracje (np. [64], [128, 64], [256, 128, 64])
   - Funkcje aktywacji: ReLU lub tanh w warstwach ukrytych, sigmoid w warstwie wyjściowej
4. **Trening**: Uczenie nadzorowane z optymalizatorem Adam i funkcją straty binary_crossentropy
5. **Klasyfikacja**: Predykcja przy użyciu funkcji sigmoid w warstwie wyjściowej

**Zalety metody**
- **Wysoka zdolność uogólniania** - sieci neuronowe dobrze radzą sobie ze złożonymi wzorcami w danych
- **Automatyczna ekstrakcja cech** - TF-IDF automatycznie identyfikuje istotne słowa i frazy
- **Skalowalność** - możliwość obsługi dużych zbiorów danych
- **Elastyczność architektury** - łatwa modyfikacja liczby warstw i neuronów 

**Wady metody**
- **Wysokie wymagania obliczeniowe** - dłuższy czas treningu w porównaniu do prostszych metod 
- **Złożoność interpretacji** - trudność w zrozumieniu, które cechy są najważniejsze 
- **Zależność od preprocessingu** - jakość danych wejściowych znacząco wpływa na wyniki 

---

**Konfiguracja programu**  
Podobnie jak poprzednio, na wstępie należało zdefiniować stałe konfiguracyjne. Konfiguracja została zaprojektowana do porównania różnych architektur sieci neuronowych pod kątem skuteczności klasyfikacji spam/ham. Eksperyment obejmuje testowanie różnych rozmiarów sieci oraz funkcji aktywacji. Najważniejsze parametry to:  
- **`MAX_FEATURES = 20000`** - Optymalny kompromis między dokładnością a wymaganiami pamięciowymi 
- **`EPOCHS = 5`** - Wystarczająca liczba epok dla zbieżności przy zachowaniu rozsądnego czasu treningu
- **`BATCH_SIZE = 128`** - Efektywny rozmiar batcha dla dużego zbioru danych 
- **`USE_PREPROCESSING = True`** - Wykorzystanie pełnego preprocessingu NLTK 

---

**Tabela porównawcza wyników dla różnych konfiguracji DNN:**
| Model | Warstwy | Aktywacja | Accuracy | Czas treningu | Czas predykcji | Poprawny spam | Fałszywe negatywy | Fałszywe pozytywne | Poprawny ham |
|-------|---------|-----------|----------|---------------|----------------|---------------|-------------------|-------------------|-------------|
| **small** | [64] | relu | **99.67%** | 34.64s | 1.32s | **65.78%** | **0.09%** | 0.24% | 33.88% |
| **medium** | [128, 64] | relu | 99.64% | 56.67s | 1.03s | **65.78%** | **0.09%** | 0.27% | 33.85% |
| **large** | [256, 128, 64] | relu | 99.61% | 99.26s | 1.17s | 65.76% | 0.12% | 0.27% | 33.85% |
| **small_tanh** | [64] | tanh | 99.64% | **31.32s** | **0.83s** | 65.77% | 0.11% | **0.25%** | **33.87%** |

**Wpływ architektury sieci na skuteczność**  
Analiza wpływu architektury sieci neuronowych na skuteczność klasyfikacji pokazała, że najprostszy z testowanych modeli – `wariant small` – osiągnął najwyższą dokładność na poziomie 99,67%. Wynik ten sugeruje, że w przypadku tego konkretnego zadania bardziej złożone i głębsze architektury nie wnoszą dodatkowych korzyści. Proste modele dysponują wystarczającą pojemnością, aby skutecznie uchwycić zależności w danych, natomiast zwiększanie liczby warstw nie tylko nie poprawia jakości, ale może wręcz prowadzić do nieznacznego przeuczenia, co potwierdzają słabsze wyniki modelu large.

**Analiza funkcji aktywacji**  
W badaniu funkcji aktywacji najlepiej wypadła funkcja `ReLU` zastosowana w modelu `small`, choć różnice względem `tanh` okazały się minimalne – odpowiednio 99,67% i 99,64% dokładności. Co ciekawe, wariant `small_tanh` zapewnił najszybszy czas zarówno treningu, jak i predykcji, co czyni go atrakcyjną alternatywą w kontekście optymalizacji wydajności.

**Wydajność czasowa**
Zgodnie z oczekiwaniami, czas treningu rósł wraz ze złożonością architektury – od 31 sekund w przypadku najprostszego modelu do 99 sekund dla najgłębszego. Czasy predykcji dla wszystkich wariantów pozostawały natomiast bardzo krótkie i mieściły się w przedziale od 0,83 do 1,32 sekundy dla całego zbioru testowego. W rezultacie to właśnie model `small` okazał się najbardziej zrównoważonym rozwiązaniem, oferując najwyższą skuteczność przy relatywnie krótkim czasie treningu.

**Jakość klasyfikacji**  
W kontekście jakości klasyfikacji wszystkie testowane architektury poradziły sobie znakomicie, generując jedynie minimalne błędy. Odsetek fałszywych negatywów wynosił zaledwie 0,09–0,12%, co oznacza, że tylko niewielka część spamu pozostawała nieodfiltrowana. Równie niski poziom fałszywych pozytywów (0,24–0,27%) wskazuje, że klasyfikatory rzadko błędnie oznaczały prawidłowe wiadomości jako spam.

---

**Porównanie z metodami z poprzednich zadań**
| Metoda | Najlepsza accuracy | Czas przetwarzania | Zalety | Wady |
|--------|-------------------|-------------------|--------|------|
| **Blacklista** | 61.83% | 2465s | Prosta, interpretowalna | Bardzo niska skuteczność |
| **LSH** | 94.50% | ~591s | Skalowalna, dobre podobieństwa | Zależna od parametrów |
| **Naive Bayes** | 99.24% | 70s | Szybki, wysoka skuteczność | Założenie niezależności |
| **DNN (small)** | **99.67%** | **~36s** | **Najwyższa dokładność**, dobre uogólnianie | Wymaga preprocessingu |

Przeprowadzone eksperymenty wykazały, że głębokie sieci neuronowe (DNN) stanowią najbardziej efektywną metodę spośród wszystkich testowanych podejść, osiągając najwyższą dokładność na poziomie `99,67%`. Wynik ten jest o `0,43%` lepszy niż w przypadku klasyfikatora Naive Bayes, co podkreśla potencjał bardziej zaawansowanych modeli w analizie tekstu. Co istotne, najlepsze rezultaty uzyskano dzięki niezwykle prostej architekturze – model składający się z jednej warstwy ukrytej i 64 neuronów okazał się najskuteczniejszy, co potwierdza, że w niektórych zadaniach dodatkowa złożoność nie przekłada się na wyższą jakość.

Wszystkie warianty DNN charakteryzowały się bardzo niskim poziomem błędów. Odsetek fałszywych negatywów utrzymywał się poniżej `0,15%`, co oznacza, że niemal wszystkie wiadomości spam były skutecznie wykrywane. Równie niski poziom fałszywych pozytywów – poniżej `0,3%` – świadczy o wysokiej precyzji klasyfikatorów w odróżnianiu prawidłowych wiadomości od spamu.

Pomimo zastosowania bardziej zaawansowanej techniki, czasy treningu i predykcji okazały się konkurencyjne względem prostszych metod. Modele DNN trenowały się relatywnie szybko, a ich czas przetwarzania podczas klasyfikacji pozostawał bardzo krótki, co czyni je praktycznym narzędziem także w systemach działających w czasie rzeczywistym.

Analiza pokazała, że model `small` z funkcją aktywacji `ReLU` stanowi rozwiązanie optymalne. Łączy on najwyższą skuteczność z dobrą wydajnością obliczeniową i prostotą implementacji. 

W porównaniu do początkowych metod, takich jak blacklisty czy LSH, DNN stanowią ogromny krok naprzód, poprawiając skuteczność odpowiednio o `37,84%` i `5,17%`. Połączenie głębokich sieci neuronowych z reprezentacją TF-IDF okazało się zatem najskuteczniejszym podejściem do klasyfikacji wiadomości e-mail, zapewniając najlepszy balans między dokładnością a wydajnością.


# TODO:
- Dodać diagramy kodu z Mermaid Chart (jest podobno jako dodatek do Visual Studio Code) Diagramy mają być jako skrypty, a nie jako obrazki