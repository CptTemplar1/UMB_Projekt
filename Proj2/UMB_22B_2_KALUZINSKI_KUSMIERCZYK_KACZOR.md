# Uczenie Maszynowe w Bezpieczeństwie
## Projekt 2
### Grupa 22B
### Autorzy: Przemysław Kałużiński, Jakub Kuśmierczyk, Michał Kaczor

### Oznaczenia matematyczne

* **$\mathcal{N}(\mu, \sigma)$** – rozkład normalny o średniej $\mu$ i odchyleniu standardowym $\sigma$
* **$Poisson(\lambda)$** – rozkład Poissona o parametrze $\lambda$
* **$Exp(\lambda)$** – rozkład wykładniczy o parametrze $\lambda$
* **TP, TN, FP, FN** – prawdziwie dodatnie, prawdziwie ujemne, fałszywie dodatnie, fałszywie ujemne
* **$\beta$** – współczynnik regresji logistycznej
* **$\beta_0$** – wyraz wolny
* **$\tau$** – próg decyzyjny
* **$H$** – entropia Shannona
* **$\rho$** – współczynnik korelacji Pearsona
* **$X$** – macierz cech (dane wejściowe)
* **$y$** – wektor etykiet ($0$ – normalny ruch, $1$ – atak)
* **$n$** – liczba próbek
* **$d$** – liczba cech
* **$\hat{\mu}$** – estymator średniej
* **$\hat{\sigma}$** – estymator odchylenia standardowego
* **$\epsilon$** – mała stała numeryczna zapobiegająca dzieleniu przez zero (typowo $10^{-10}$)
* **$D$** – ramka danych (dataframe)
* **$D_{Label}$** – kolumna etykiet w ramce danych
* **$D_{DestPort}$** – kolumna portów docelowych w ramce danych
* **$L$** – wektor etykiet (tekstowych lub numerycznych)
* **$L_{norm}$** – etykieta klasy normalnej (wartość oznaczająca normalny ruch)
* **$T_{type}$** – kolumna typu ataku w ramce danych
* **$T_{DDoS}$** – wartość etykiety oznaczająca atak DDoS
* **$T_{Port}$** – wartość etykiety oznaczająca atak Port Scan

### Notacja funkcji w algorytmach

* **$\mathbb{E}[X]$** – wartość oczekiwana (średnia) wektora $X$
* **$\sigma[X]$** – odchylenie standardowe wektora $X$
* **$Var[X]$** – wariancja wektora $X$
* **$Med[X]$** – mediana wektora $X$
* **$\oplus$** – konkatenacja (łączenie) struktur danych
* **$\pi$** – permutacja losowa elementów
* **$\pi(X, y)$** – aplikacja permutacji losowej do danych $X$ i etykiet $y$
* **$sort(X, key)$** – sortowanie zbioru $X$ według klucza
* **$argsort(X)$** – zwraca indeksy sortujące wektor $X$ rosnąco
* **$clip(x, a, b)$** – obcięcie wartości $x$ do przedziału $[a, b]$
* **$⊮(\cdot)$** – funkcja wskaźnikowa (1 gdy warunek prawdziwy, 0 w przeciwnym przypadku)
* **$\triangleright$** – symbol komentarza w algorytmach
* **$M(y, \hat{y})$** – macierz pomyłek zwracająca (TP, TN, FP, FN) dla etykiet $y$ i predykcji $\hat{y}$
* **$H(\{p_i\})$** – entropia Shannona: $H = - \sum_i p_i \log_2(p_i)$
* **$\arg \min_x f(x)$** i **$\arg \max_x f(x)$** – argument minimalizujący/maksymalizujący funkcję $f$
* **$x^*$** – notacja oznaczająca wartość optymalną (po optymalizacji)
* **$0_{n \times m}$** – macierz zer o wymiarach $n \times m$
* **$0_n$** – wektor zer o długości $n$
* **unique(S)** – zwraca zbiór unikalnych elementów ze zbioru $S$
* **read_csv(f)** – wczytuje dane z pliku CSV o ścieżce $f$
* **is_numeric(X)** – predykat zwracający prawdę gdy $X$ jest typu numerycznego
* **num_cols(D)** – zwraca liczbę kolumn w macierzy/ramce danych $D$
* **sample(S, k)** – zwraca losowy podzbiór $k$ elementów ze zbioru $S$
* **stratified_split(X, y, p)** – podział ze stratyfikacją w proporcji $p : (1 - p)$
* **$|$** – separator w definicjach matematycznych (odpowiednik "gdzie")

### Operatory algorytmiczne

* **$\leftarrow$** – operator przypisania (strzałka w lewo): przypisanie wartości do zmiennej
* **$\sim$** – operator losowania: $x \sim \mathcal{N}(\mu, \sigma)$ oznacza wylosowanie wartości $x$ z rozkładu normalnego
* **$\wedge$** – koniunkcja logiczna (AND): $a \wedge b$ jest prawdziwe gdy oba warunki są spełnione
* **$\vee$** – alternatywa logiczna (OR): $a \vee b$ jest prawdziwe gdy przynajmniej jeden warunek jest spełniony
* **$\setminus$** – różnica zbiorów: $A \setminus B$ zawiera elementy z $A$ nieobecne w $B$
* **$\in$** – przynależność do zbioru: $x \in S$ oznacza, że element $x$ należy do zbioru $S$
* **$\exists$** – kwantyfikator egzystencjalny: istnieje
* **$\forall$** – kwantyfikator uniwersalny: dla każdego

### Implementacja projektu
**Plik data_generation.py**

Plik ten odpowiada za generowanie danych syntetycznych dla Zadania 1 i Zadania 2 oraz wczytywanie danych rzeczywistych dla Zadania 3. Implementuje logikę opisaną w Algorytmach 1 i 2 z dokumentacji projektu.

**Zmienne globalne i importy**
- Importowane są biblioteki `numpy`, `pandas` oraz `os`.
- Kod korzysta z generatora liczb losowych `numpy`, co pozwala na reprodukowalność wyników dzięki ustawieniu ziarna losowości (`seed`).

**Kod:**
```python
import numpy as np
import pandas as pd
import os
```

**1. Funkcja `generate_ideal_data`**

**Wejście:**
- `n_norm` (int, domyślnie 800) – liczba próbek klasy normalnej.
- `n_attack` (int, domyślnie 200) – liczba próbek klasy ataku.
- `seed` (int, domyślnie 42) – ziarno losowości dla powtarzalności wyników.

**Wyjście:**
- `df` (pandas.DataFrame) – ramka danych zawierająca wygenerowane cechy oraz etykietę `Target`.
- `feature_names` (list) – lista nazw 7 wygenerowanych cech.

**Opis:**

Funkcja realizuje **Algorytm 1** (Zadanie 1: Eksperyment z danymi idealnymi). Generuje syntetyczny zbiór danych, gdzie klasa normalna i ataki są wyraźnie odseparowane.
- Dla ruchu normalnego losuje wartości z rozkładów (Normalny, Poisson, Wykładniczy) o parametrach określonych w treści zadania (np. pakiety/s: średnia=50, odchylenie=15).
- Dla ataków używa przesuniętych parametrów (np. pakiety/s: średnia=250, odchylenie=30).
- Łączy dane w jeden zbiór, dodaje etykiety (0 – normalny, 1 – atak) i dokonuje permutacji (wymieszania) próbek.

**Kod:**
```python
def generate_ideal_data(n_norm=800, n_attack=200, seed=42):
    np.random.seed(seed)
    
    # Parametry (zgodne z treścią zadania)
    X_norm = np.zeros((n_norm, 7))
    X_norm[:, 0] = np.random.normal(50, 15, n_norm)              # Pakiety/s
    X_norm[:, 1] = np.random.normal(800, 200, n_norm)            # Rozmiar
    X_norm[:, 2] = np.random.normal(2.5, 0.5, n_norm)            # Entropia
    X_norm[:, 3] = np.clip(np.random.normal(0.2, 0.05, n_norm), 0, 1) # SYN
    X_norm[:, 4] = np.random.poisson(5, n_norm) + 1              # IP
    X_norm[:, 5] = np.random.exponential(30, n_norm)             # Czas
    X_norm[:, 6] = np.random.poisson(2, n_norm)                  # Powtórzenia
    y_norm = np.zeros(n_norm)

    X_attack = np.zeros((n_attack, 7))
    X_attack[:, 0] = np.random.normal(250, 30, n_attack)
    X_attack[:, 1] = np.random.normal(300, 100, n_attack)
    X_attack[:, 2] = np.random.normal(4.0, 0.3, n_attack)
    X_attack[:, 3] = np.clip(np.random.normal(0.8, 0.05, n_attack), 0, 1)
    X_attack[:, 4] = np.random.poisson(50, n_attack) + 1
    X_attack[:, 5] = np.random.exponential(2, n_attack)
    X_attack[:, 6] = np.random.poisson(20, n_attack)
    y_attack = np.ones(n_attack)

    feature_names = ['packetspersec', 'avgpacketsize', 'portentropy', 'synratio', 'uniquedstips', 'connectionduration', 'repeatedconnections']
    
    X = np.vstack([X_norm, X_attack])
    y = np.hstack([y_norm, y_attack])
    
    df = pd.DataFrame(X, columns=feature_names)
    df['Target'] = y
    
    # Mieszanie (permutacja)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df, feature_names
```

**2. Funkcja `generate_realistic_data`**

**Wejście:**
- `n_norm` (int, domyślnie 950) – liczba próbek klasy normalnej.
- `n_obv` (int, domyślnie 20) – liczba ataków oczywistych.
- `n_med` (int, domyślnie 15) – liczba ataków średnio subtelnych.
- `n_sub` (int, domyślnie 15) – liczba ataków bardzo subtelnych.
- `seed` (int, domyślnie 42) – ziarno losowości.

**Wyjście:**
- `df` (pandas.DataFrame) – ramka danych z cechami, etykietą `Target` oraz typem ataku `Type`.
- `feature_names` (list) – lista nazw cech.

**Opis:**

Funkcja realizuje **Algorytm 2** (Zadanie 2: Eksperyment z danymi realistycznymi). Tworzy silnie niezbalansowany zbiór danych (proporcja 950:50) symulujący rzeczywiste warunki sieciowe.
- Normalny ruch generowany jest jak w Zadaniu 1.
- Ataki są generowane w trzech kategoriach trudności, poprzez przesunięcie średniej rozkładu normalnego ruchu o wielokrotność odchylenia standardowego (k*sigma):
    - Ataki oczywiste: przesunięcie o 4 odchylenia standardowe.
    - Ataki średnie: przesunięcie o 2 odchylenia standardowe.
    - Ataki subtelne: przesunięcie o 1 odchylenie standardowe.
- Funkcja pomocnicza `add_attacks` obsługuje logikę przesunięć w zależności od tego, czy cecha rośnie, czy maleje podczas ataku.

**Kod:**
```python
def generate_realistic_data(n_norm=950, n_obv=20, n_med=15, n_sub=15, seed=42):
    np.random.seed(seed)
    
    mu_norm = np.array([50, 800, 2.5, 0.2, 5, 30, 2])
    sigma_norm = np.array([15, 200, 0.5, 0.05, 0, 0, 0]) 
    
    feature_names = ['packetspersec', 'avgpacketsize', 'portentropy', 'synratio', 'uniquedstips', 'connectionduration', 'repeatedconnections']
    data_list = []

    # 1. Normalny
    for _ in range(n_norm):
        row = [
            np.random.normal(mu_norm[0], sigma_norm[0]),
            np.random.normal(mu_norm[1], sigma_norm[1]),
            np.random.normal(mu_norm[2], sigma_norm[2]),
            np.clip(np.random.normal(mu_norm[3], sigma_norm[3]), 0, 1),
            np.random.poisson(mu_norm[4]) + 1,
            np.random.exponential(mu_norm[5]),
            np.random.poisson(mu_norm[6])
        ]
        data_list.append(row + [0, 'Normal'])
            
    # Helper dla ataków
    def add_attacks(n, k, type_name):
        for _ in range(n):
            row = [
                np.random.normal(mu_norm[0] + k*sigma_norm[0], sigma_norm[0]),
                np.random.normal(mu_norm[1] - k*sigma_norm[1], sigma_norm[1]),
                np.random.normal(mu_norm[2] + k*sigma_norm[2], sigma_norm[2]),
                np.clip(np.random.normal(mu_norm[3] + k*sigma_norm[3], sigma_norm[3]), 0, 1),
                np.random.poisson(mu_norm[4] + k*10) + 1,
                np.random.exponential(mu_norm[5] / k),
                np.random.poisson(mu_norm[6] + k*5)
            ]
            data_list.append(row + [1, type_name])

    add_attacks(n_obv, 4, 'Obvious')
    add_attacks(n_med, 2, 'Medium')
    add_attacks(n_sub, 1, 'Subtle')
    
    df = pd.DataFrame(data_list, columns=feature_names + ['Target', 'Type'])
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df, feature_names
```

**3. Funkcja `load_cicids_data`**

**Wejście:**
- `data_dir` (str) – ścieżka do katalogu z plikami CSV (CICIDS2017).
- `sample_size` (int, domyślnie 50000) – liczba próbek pobieranych z każdego pliku (dla wydajności).
- `seed` (int, domyślnie 42) – ziarno losowości.

**Wyjście:**
- `full_df` (pandas.DataFrame) – połączona i wylosowana próbka danych rzeczywistych.

**Opis:**

Funkcja obsługuje wczytywanie danych rzeczywistych wymaganych w **Zadaniu 3**.
- Definiuje mapowanie plików dla ruchu normalnego (`Monday`), ataków DDoS (`Friday-DDoS`) i PortScan (`Friday-PortScan`).
- Wczytuje pliki CSV, usuwa zbędne spacje z nazw kolumn.
- Wykonuje losowe próbkowanie (`sample`), aby zmniejszyć rozmiar danych do poziomu umożliwiającego szybkie obliczenia, co jest zgodne z zaleceniami optymalizacji wydajności.
- Łączy dane w jedną ramkę (`pd.concat`).

**Kod:**
```python
def load_cicids_data(data_dir, sample_size=50000, seed=42):

    files = {
        'normal': 'Monday-WorkingHours.pcap_ISCX.csv',
        'ddos': 'Friday-WorkingHours-Afternoon-DDoS.pcap_ISCX.csv',
        'portscan': 'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv'
    }
    
    dfs = []
    for k, fname in files.items():
        path = os.path.join(data_dir, fname)
        if os.path.exists(path):
            print(f"Wczytywanie: {fname}")
            try:
                temp_df = pd.read_csv(path)
                temp_df.columns = temp_df.columns.str.strip() # Usuwanie spacji z nazw kolumn
                # Sampling dla wydajności
                if len(temp_df) > sample_size:
                    temp_df = temp_df.sample(n=sample_size, random_state=seed)
                dfs.append(temp_df)
            except Exception as e:
                print(f"Błąd wczytywania {fname}: {e}")
        else:
            print(f"Ostrzeżenie: Plik {path} nie istnieje.")
            
    if not dfs:
        raise FileNotFoundError("Nie znaleziono plików danych.")
        
    full_df = pd.concat(dfs, ignore_index=True)
    return full_df.sample(frac=1, random_state=seed).reset_index(drop=True)
```

---

**Plik feature_engineering.py**

Ten plik zawiera funkcje odpowiedzialne za czyszczenie, transformację i normalizację danych rzeczywistych (CICIDS2017), realizując wytyczne z Algorytmów 5 i 6.

**Zmienne globalne i importy**
Podobnie jak wcześniej, importowane są biblioteki `numpy` oraz `pandas`. Dodatkowo importowana jest klasa `StandardScaler` z `sklearn.preprocessing` do normalizacji cech. 

**Kod:**
```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
```

**1. Funkcja `clean_cicids_data`**

**Wejście:**
- `df` (pandas.DataFrame) – surowe dane wczytane z plików CSV.

**Wyjście:**
- `df_clean` (pandas.DataFrame) – oczyszczona ramka danych.

**Opis:**

Realizuje **Algorytm 5** (Czyszczenie i przygotowanie danych rzeczywistych).
- Usuwa kolumny identyfikacyjne (np. 'Flow ID', 'Source IP', 'Timestamp'), które nie niosą informacji predykcyjnej. Pozostawia `Destination Port` potrzebny do dalszych obliczeń.
- Zamienia wartości nieskończone (`inf`) na `NaN`, a następnie uzupełnia braki danych medianą kolumny.
- Usuwa kolumny o zerowej lub bardzo niskiej wariancji (< 1e-10), które są bezużyteczne dla modelu.

**Kod:**
```python
def clean_cicids_data(df):
    # 1. Usuwanie kolumn ID
    cols_to_remove = ['Flow ID', 'Source IP', 'Destination IP', 'Timestamp', 'Source Port', 'Socket']
    # Zachowujemy 'Destination Port' bo jest potrzebny do algorytmów
    df_clean = df.drop(columns=[c for c in cols_to_remove if c in df.columns], errors='ignore')
    
    # 2. Obsługa nieskończoności i NaN
    df_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_clean.fillna(df_clean.median(numeric_only=True), inplace=True)
    
    # 3. Usuwanie zerowej wariancji (proste podejście)
    numeric_cols = df_clean.select_dtypes(include=np.number).columns
    var = df_clean[numeric_cols].var()
    drop_cols = var[var < 1e-10].index
    df_clean.drop(columns=drop_cols, inplace=True)
    
    return df_clean
```

**2. Funkcja `engineer_features_cicids`**

**Wejście:**
- `df` (pandas.DataFrame) – oczyszczona ramka danych (z poprzedniego kroku).

**Wyjście:**
- `features` (pandas.DataFrame) – ramka zawierająca tylko 7 wyselekcjonowanych cech bazowych.
- `y` (pandas.Series) – binarne etykiety (0 - BENIGN, 1 - atak).
- `df['Label']` (pandas.Series) – oryginalne etykiety tekstowe (do analizy typów ataków).

**Opis:**

Realizuje **Algorytm 6** (Inżynieria cech dla danych rzeczywistych). Tworzy zestaw 7 cech odpowiadających tym z danych syntetycznych:
- Mapuje bezpośrednio dostępne kolumny, np. `Flow Packets/s` na `packetspersec`.
- Oblicza `synratio` jako stosunek flag SYN do całkowitej liczby pakietów.
- Przelicza czas trwania (`Flow Duration`) z mikrosekund na sekundy.
- **Ważne:** Dla cech trudnych obliczeniowo (wymagających agregacji czasowej na milionach rekordów), funkcja stosuje aproksymacje oparte na dostępnych statystykach przepływu (np. używa `Bwd Header Length` jako proxy dla entropii czy `Fwd IAT Max` dla powtórzeń), co pozwala na wykonanie kodu w rozsądnym czasie na standardowym sprzęcie.
- Konwertuje etykiety tekstowe na binarne (wszystko co nie jest 'BENIGN' staje się klasą 1).

**Kod:**
```python
def engineer_features_cicids(df):
    features = pd.DataFrame(index=df.index)
    
    # Bezpośrednie mapowanie
    features['packetspersec'] = df['Flow Packets/s'] if 'Flow Packets/s' in df else 0
    features['avgpacketsize'] = df['Average Packet Size'] if 'Average Packet Size' in df else 0
    features['connectionduration'] = (df['Flow Duration'] / 1e6) if 'Flow Duration' in df else 0
    
    # SYN Ratio
    total = df['Total Fwd Packets'] + df['Total Backward Packets'] + 1e-10
    syn_counts = df['SYN Flag Count'] if 'SYN Flag Count' in df else 0
    features['synratio'] = np.clip(syn_counts / total, 0, 1)
    
    # Aproksymacje cech trudnych obliczeniowo (bez pełnej agregacji IP)
    # Port Entropy -> Bwd Header Length jako proxy złożoności
    features['portentropy'] = df['Bwd Header Length'] if 'Bwd Header Length' in df else 0
    
    # Unique IPs -> Aproksymacja np. przez Total Length of Bwd Packets
    features['uniquedstips'] = df['Total Length of Bwd Packets'] if 'Total Length of Bwd Packets' in df else 0
    
    # Repeated Connections -> Aproksymacja przez Fwd IAT Max
    features['repeatedconnections'] = df['Fwd IAT Max'] if 'Fwd IAT Max' in df else 0
    
    # Etykiety
    y = (df['Label'] != 'BENIGN').astype(int)
    
    feature_names = ['packetspersec', 'avgpacketsize', 'portentropy', 'synratio', 
                     'uniquedstips', 'connectionduration', 'repeatedconnections']
                     
    # Sprzątanie finalne
    features.replace([np.inf, -np.inf], 0, inplace=True)
    features.fillna(0, inplace=True)
    
    return features[feature_names], y, df['Label'] # Zwracamy też oryginalne etykiety tekstowe
```

**3. Funkcja `normalize_data`**

**Wejście:**
- `X_train` (array-like) – zbiór treningowy.
- `X_test` (array-like) – zbiór testowy.
- `X_val` (array-like, opcjonalnie) – zbiór walidacyjny.

**Wyjście:**
- Przeskalowane macierze (`X_train_scaled`, `X_test_scaled`, ew. `X_val_scaled`) oraz obiekt `scaler`.

**Opis:**

Funkcja odpowiada za standaryzację cech (z-score normalization). Kluczowym elementem implementacji jest to, że parametry normalizacji (średnia, odchylenie) są obliczane **wyłącznie na zbiorze treningowym** (`fit_transform`), a następnie aplikowane do zbioru testowego/walidacyjnego (`transform`). Zapobiega to wyciekowi danych (data leakage), co jest kluczowe w uczeniu maszynowym.

**Kod:**
```python
def normalize_data(X_train, X_test, X_val=None):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    if X_val is not None:
        X_val_scaled = scaler.transform(X_val)
        return X_train_scaled, X_test_scaled, X_val_scaled, scaler
        
    return X_train_scaled, X_test_scaled, scaler
```

---

**Plik model_training.py**

Plik zawiera logikę trenowania modelu regresji logistycznej, optymalizacji progu decyzyjnego oraz ewaluacji wyników.

**Zmienne globalne i importy**
Importowane są biblioteki `numpy`, `pandas` oraz klasy i funkcje z `sklearn` potrzebne do trenowania modelu i obliczania metryk ewaluacyjnych.

**Kod:**
```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
```

**1. Funkcja `train_model`**

**Wejście:**
- `X_train`, `y_train` (array-like) – dane treningowe.
- `class_weight` (dict/str, domyślnie None) – wagi klas (np. 'balanced' dla Zadania 2).
- `C` (float, domyślnie 1.0) – siła regularyzacji (odwrotność lambda).
- `max_iter` (int, domyślnie 1000) – maksymalna liczba iteracji solvera.

**Wyjście:**
- `model` (sklearn.linear_model.LogisticRegression) – wytrenowany obiekt modelu.

**Opis:**

Wrapper na klasę `LogisticRegression` z biblioteki scikit-learn. Implementuje minimalizację funkcji kosztu log-loss z regularyzacją L2. Obsługuje parametr `class_weight`, który jest kluczowy w Zadaniu 2 do radzenia sobie z niezbalansowanymi danymi.

**Kod:**
```python
def train_model(X_train, y_train, class_weight=None, C=1.0, max_iter=1000):
    model = LogisticRegression(penalty='l2', C=C, class_weight=class_weight, 
                               solver='lbfgs', max_iter=max_iter, random_state=42)
    model.fit(X_train, y_train)
    return model
```

**2. Funkcja `optimize_threshold`**

**Wejście:**
- `model` (object) – wytrenowany model.
- `X_test`, `y_test` (array-like) – dane testowe.
- `wFN` (int, domyślnie 100) – koszt błędu Fałszywie Ujemnego (False Negative).
- `wFP` (int, domyślnie 1) – koszt błędu Fałszywie Dodatniego (False Positive).

**Wyjście:**
- `best_tau` (float) – optymalny próg.
- `costs` (list) – lista kosztów dla każdego progu.
- `thresholds` (array) – sprawdzone progi.

**Opis:**

Realizuje **Algorytm 3** (Optymalizacja progu decyzyjnego).
- Iteruje przez możliwe progi decyzyjne od 0.01 do 0.99.
- Dla każdego progu oblicza macierz pomyłek i całkowity koszt według wzoru: 100 * FN + 1 * FP.
- Znajduje próg optymalny, który minimalizuje tę funkcję kosztu. Jest to kluczowe dla minimalizacji ryzyka przepuszczenia ataku.

**Kod:**
```python
def optimize_threshold(model, X_test, y_test, wFN=100, wFP=1):
    probs = model.predict_proba(X_test)[:, 1]
    thresholds = np.arange(0.01, 1.00, 0.01)
    costs = []
    
    best_c = float('inf')
    best_tau = 0.5
    
    for tau in thresholds:
        y_pred = (probs >= tau).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        cost = wFN * fn + wFP * fp
        costs.append(cost)
        
        if cost < best_c:
            best_c = cost
            best_tau = tau
            
    return best_tau, costs, thresholds
```

**3. Funkcja `evaluate_model`**

**Wejście:**
- `model` (object) – wytrenowany model.
- `X_test`, `y_test` (array-like) – dane testowe.
- `threshold` (float, domyślnie 0.5) – próg decyzyjny.

**Wyjście:**
- `metrics` (dict) – słownik z wynikami (Accuracy, Precision, Recall, F1, AUC, TP, TN, FP, FN).
- `y_pred` (array) – binarne decyzje modelu.
- `probs` (array) – prawdopodobieństwa klasy pozytywnej.

**Opis:**

Funkcja agregująca obliczenia wszystkich wymaganych metryk ewaluacyjnych opisanych w **Dodatku A** dokumentacji.
- Oblicza prawdopodobieństwa P(y=1|x).
- Aplikuje próg decyzyjny, aby uzyskać klasyfikację binarną.
- Zwraca słownik zawierający Accuracy, Precision, Recall, F1 oraz składowe macierzy pomyłek.
- Oblicza również AUC (Area Under Curve) na podstawie krzywej ROC.

**Kod:**
```python
def evaluate_model(model, X_test, y_test, threshold=0.5):
    probs = model.predict_proba(X_test)[:, 1]
    y_pred = (probs >= threshold).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'F1': f1_score(y_test, y_pred, zero_division=0),
        'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn
    }
    
    fpr, tpr, _ = roc_curve(y_test, probs)
    metrics['AUC'] = auc(fpr, tpr)
    metrics['FPR'] = fpr
    metrics['TPR'] = tpr
    
    return metrics, y_pred, probs
```

**4. Funkcja `analyze_errors_task3`**

**Wejście:**
- `X_test_raw` (array-like) – nieznormalizowane dane testowe.
- `y_test`, `y_pred` (array-like) – etykiety prawdziwe i przewidziane.
- `y_prob` (array-like) – prawdopodobieństwa.
- `feature_names` (list) – nazwy cech.

**Wyjście:**
- `X_df.iloc[fn_idx]` (DataFrame) – próbki błędnie sklasyfikowane jako normalne (FN).
- `X_df.iloc[fp_idx]` (DataFrame) – próbki błędnie sklasyfikowane jako atak (FP).
- `fn_idx`, `fp_idx` (array) – indeksy błędów.

**Opis:**

Pomocnicza funkcja realizująca część **Algorytmu 7** (Analiza błędów klasyfikacji). Służy do wyodrębnienia konkretnych przypadków błędów (False Negatives i False Positives), co pozwala na ich późniejszą ręczną analizę i zrozumienie, dlaczego model się pomylił (np. czy atak był zbyt subtelny).

**Kod:**
```python
def analyze_errors_task3(X_test_raw, y_test, y_pred, y_prob, feature_names):
    X_df = pd.DataFrame(X_test_raw, columns=feature_names)
    
    # Indeksy błędów
    fn_idx = np.where((y_test == 1) & (y_pred == 0))[0]
    fp_idx = np.where((y_test == 0) & (y_pred == 1))[0]
    
    return X_df.iloc[fn_idx], X_df.iloc[fp_idx], fn_idx, fp_idx
```

---

**Plik visualization.py**

Plik zawiera funkcje wizualizacyjne oparte na bibliotekach `matplotlib` i `seaborn`, służące do generowania wykresów wymaganych w raporcie końcowym.

**Zmienne globalne i importy**
Importowane są biblioteki `matplotlib.pyplot`, `seaborn`, `pandas`, `numpy` oraz funkcje z `sklearn.metrics` i `sklearn.model_selection`. Ich użycie jest kluczowe do tworzenia wykresów takich jak rozkłady cech, macierze korelacji, krzywe ROC, itp.

**Kod:**
```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import learning_curve
```

**1. Funkcja `plot_distributions`**

**Wejście:**
- `df` (pandas.DataFrame) – dane.
- `feature_names` (list) – nazwy cech do wyrysowania.
- `target_col`, `title_prefix` – parametry konfiguracyjne.

**Wyjście:**
- Wykresy (okno matplotlib).

**Opis:**

Rysuje rozkłady gęstości (KDE) dla cech, porównując klasę normalną (niebieski) i ataki (czerwony). Pozwala ocenić stopień separacji klas i trudność zadania klasyfikacji.

**Kod:**
```python
def plot_distributions(df, feature_names, target_col='Target', title_prefix=''):
    n_features = len(feature_names)
    n_cols = 2
    n_rows = (n_features + 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten()
    
    for i, col in enumerate(feature_names):
        if i >= len(axes): break
        # Rysowanie tylko jeśli są dane dla obu klas
        if df[df[target_col]==0].shape[0] > 0:
            sns.kdeplot(data=df[df[target_col]==0], x=col, fill=True, color='blue', label='Normal', ax=axes[i], warn_singular=False)
        if df[df[target_col]==1].shape[0] > 0:
            sns.kdeplot(data=df[df[target_col]==1], x=col, fill=True, color='red', label='Attack', ax=axes[i], warn_singular=False)
            
        axes[i].set_title(f'{title_prefix} Rozkład: {col}')
        axes[i].legend()
    
    plt.tight_layout()
    plt.show()
```

**2. Funkcja `plot_correlation_matrix`**

**Wejście:**
- `df` (pandas.DataFrame) – dane.
- `feature_names` (list) – nazwy cech.

**Wyjście:**
- Wykres heatmapy.

**Opis:**

Oblicza i wizualizuje macierz korelacji Pearsona między cechami numerycznymi. Używa heatmapy do pokazania siły zależności liniowych (od -1 do 1).

**Kod:**
```python
def plot_correlation_matrix(df, feature_names):
    plt.figure(figsize=(10, 8))
    # Obliczamy korelację tylko dla cech numerycznych
    corr = df[feature_names].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool)) # Opcjonalnie: maskowanie górnego trójkąta
    
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    plt.title("Macierz korelacji Pearsona")
    plt.show()
```

**3. Funkcja `plot_betas`**

**Wejście:**
- `model` (object) – model regresji.
- `feature_names` (list) – nazwy cech.
- `title` (str) – tytuł wykresu.

**Wyjście:**
- Wykres słupkowy.

**Opis:**

Wizualizuje globalne znaczenie cech poprzez wykres współczynników beta modelu. Słupki są kolorowane wg znaku (czerwony dla dodatniego wpływu na ryzyko ataku, niebieski dla ujemnego) i sortowane wg wartości bezwzględnej.

**Kod:**
```python
def plot_betas(model, feature_names, title='Współczynniki Beta (Globalne znaczenie cech)'):
    if hasattr(model, 'coef_'):
        beta = model.coef_[0]
    else:
        print("Model nie posiada atrybutu coef_ (nie jest liniowy).")
        return

    df = pd.DataFrame({'Feature': feature_names, 'Beta': beta, 'Abs': np.abs(beta)})
    df = df.sort_values('Abs', ascending=True)
    
    plt.figure(figsize=(10, 6))
    colors = ['crimson' if x > 0 else 'royalblue' for x in df['Beta']]
    bars = plt.barh(df['Feature'], df['Beta'], color=colors)
    plt.bar_label(bars, fmt='%.2f', padding=3)
    plt.title(title)
    plt.xlabel('Wartość współczynnika (Log-odds)')
    plt.grid(True, alpha=0.3)
    plt.show()
```

**4. Funkcja `plot_feature_impact`**

**Wejście:**
- `model` (object) – model.
- `X_sample` (array) – próbki danych.
- `feature_names` (list) – nazwy cech.
- `sample_idx` (int) – indeks próbki do analizy.

**Wyjście:**
- Wykres słupkowy.

**Opis:**

Zapewnia lokalną interpretowalność modelu dla konkretnej próbki. Pokazuje wkład każdej cechy w decyzję, obliczony jako iloczyn wartości cechy i jej współczynnika (beta * x).

**Kod:**
```python
def plot_feature_impact(model, X_sample, feature_names, sample_idx=0, prediction_prob=None):
    if not hasattr(model, 'coef_'): return
    
    beta = model.coef_[0]
    x_values = X_sample[sample_idx]
    impact = beta * x_values
    
    df = pd.DataFrame({'Feature': feature_names, 'Impact': impact})
    df = df.sort_values('Impact', ascending=True)
    
    plt.figure(figsize=(10, 5))
    colors = ['crimson' if x > 0 else 'royalblue' for x in df['Impact']]
    plt.barh(df['Feature'], df['Impact'], color=colors)
    
    title = f'Wpływ cech dla próbki #{sample_idx}'
    if prediction_prob is not None:
        title += f' (P(Atak) = {prediction_prob:.4f})'
        
    plt.title(title)
    plt.xlabel('Wkład w decyzję (Beta * Wartość Cechy)')
    plt.show()
```

**5. Funkcja `plot_confusion_matrix`**

**Wejście:**
- `y_true`, `y_pred` (array) – etykiety.
- `title` (str) – tytuł.

**Wyjście:**
- Wykres heatmapy.

**Opis:**

Rysuje macierz pomyłek z adnotacjami zawierającymi zarówno liczby bezwzględne, jak i procenty. Pozwala szybko ocenić liczbę błędów FP i FN.

**Kod:**
```python
def plot_confusion_matrix(y_true, y_pred, title='Macierz Pomyłek'):
    cm = confusion_matrix(y_true, y_pred)
    
    # Dodanie etykiet z liczbami i procentami
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
                
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', 
                xticklabels=['Pred: Normal', 'Pred: Atak'], 
                yticklabels=['Real: Normal', 'Real: Atak'])
    plt.title(title)
    plt.ylabel('Rzeczywista klasa')
    plt.xlabel('Przewidywana klasa')
    plt.show()
```

**6. Funkcja `plot_roc_curve`**

**Wejście:**
- `metrics` (dict) – słownik zawierający 'FPR', 'TPR', 'AUC'.
- `title` (str) – tytuł.

**Wyjście:**
- Wykres liniowy.

**Opis:**

Rysuje krzywą ROC (Receiver Operating Characteristic) i wypisuje wartość AUC w legendzie. Jest to standardowa metoda oceny klasyfikatorów binarnych.

**Kod:**
```python
def plot_roc_curve(metrics, title='Krzywa ROC'):
    plt.figure(figsize=(8, 6))
    plt.plot(metrics['FPR'], metrics['TPR'], color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {metrics["AUC"]:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.show()
```

**7. Funkcja `plot_task2_cost`**

**Wejście:**
- `thresholds` (array) – wektor progów.
- `costs` (list) – obliczone koszty.
- `opt_tau`, `opt_cost` – parametry optymalne.

**Wyjście:**
- Wykres liniowy z zaznaczonym punktem.

**Opis:**

Specyficzna dla Zadania 2 wizualizacja funkcji kosztu w zależności od progu decyzyjnego. Czerwonym punktem zaznacza znalezione minimum globalne (próg optymalny).

**Kod:**
```python
def plot_task2_cost(thresholds, costs, opt_tau, opt_cost):
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, costs, label='Funkcja kosztu', color='purple')
    plt.scatter(opt_tau, opt_cost, color='red', s=100, zorder=5, label=f'Min (tau={opt_tau:.2f})')
    plt.title('Optymalizacja progu decyzyjnego')
    plt.xlabel('Próg decyzyjny (tau)')
    plt.ylabel('Całkowity Koszt (100*FN + 1*FP)')
    plt.legend()
    plt.grid(True)
    plt.show()
```

**8. Funkcja `plot_learning_curve_manual`**

**Wejście:**
- `train_sizes` (list) – rozmiary zbiorów treningowych.
- `train_scores`, `val_scores` (list) – wyniki dokładności.

**Wyjście:**
- Wykres liniowy.

**Opis:**

Rysuje krzywą uczenia dla Zadania 3, pokazując jak zmienia się dokładność (Accuracy) na zbiorze treningowym i walidacyjnym wraz ze wzrostem liczby danych. Pomaga ocenić czy model jest przeuczony (overfitting) lub niedouczony (underfitting).

**Kod:**
```python
def plot_learning_curve_manual(train_sizes, train_scores, val_scores):
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, val_scores, 'o-', color="g", label="Validation score")
    plt.title("Krzywa uczenia")
    plt.xlabel("Liczba próbek treningowych")
    plt.ylabel("Dokładność (Accuracy)")
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()
```

### Zadanie 1: EKSPERYMENT Z DANYMI IDEALNYMI

Celem pierwszego zadania było zbadanie działania modelu regresji logistycznej w warunkach laboratoryjnych, na syntetycznym zbiorze danych charakteryzującym się wyraźną separacją klas. Wygenerowano zbiór 1000 próbek (800 normalnych, 200 ataków) opisanych 7 cechami numerycznymi (m.in. pakiety/s, entropia portów), których wartości pochodziły ze znanych rozkładów statystycznych (Normalny, Poissona, Wykładniczy). Rozwiązanie polegało na podziale danych, ich normalizacji (Z-score) oraz wytrenowaniu klasyfikatora z regularyzacją L2. Eksperyment służył jako punkt odniesienia (baseline) do oceny skuteczności detekcji, interpretowalności współczynników $\beta$ oraz analizy krzywej ROC w idealnym środowisku.

#### Wyniki

# Wykorzystanie Środowiska Jupyter Notebook

Do realizacji eksperymentów i uruchomienia przygotowanych modułów programu wykorzystano środowisko **Jupyter Notebook**. Jest to narzędzie umożliwiające tworzenie interaktywnych dokumentów, które łączą w sobie kod wykonywalny, wizualizacje oraz tekst opisowy.

Zasada działania notebooków opiera się na architekturze klient-serwer, gdzie kod jest wykonywany przez **jądro (kernel)**, a wyniki są natychmiast prezentowane użytkownikowi. Kluczową funkcjonalnością tego środowiska jest podział kodu na **komórki (cells)**. Umożliwia to:
* **Uruchamianie pojedynczych fragmentów kodu:** Użytkownik może wykonywać kod sekwencyjnie lub wybiórczo, co pozwala na szybkie testowanie poszczególnych funkcji bez konieczności przeładowywania całego programu.
* **Zachowanie stanu:** Zmienne i obiekty utworzone w jednej komórce są przechowywane w pamięci i dostępne dla innych komórek w ramach tej samej sesji.
* **Bezpośrednią wizualizację:** Wykresy i tabele (np. z bibliotek `matplotlib` czy `pandas`) są renderowane bezpośrednio pod kodem generującym.

Zgodnie ze strukturą projektu, dla każdego z zadań utworzono dedykowany notebook:
1.  `zadaniel_dane_idealne.ipynb`
2.  `zadanie2_dane_realistyczne.ipynb`
3.  `zadanie3_dane_rzeczywiste.ipynb`

Takie podejście pozwoliło na zaimportowanie wspólnych funkcji z plików źródłowych (`src/`), a następnie uruchomienie ich z unikalnymi parametrami konfiguracyjnymi (takimi jak proporcje klas, typy rozkładów czy ścieżki do plików) wymaganymi specyficznie dla każdego z trzech eksperymentów.

**Cell 1: Importy**
```python
import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Dodanie katalogu src do ścieżki
sys.path.append(os.path.abspath(os.path.join('..', 'src')))

from data_generation import generate_ideal_data
from feature_engineering import normalize_data
from model_training import train_model, evaluate_model
from visualization import plot_distributions, plot_betas, plot_confusion_matrix, plot_roc_curve, plot_correlation_matrix, plot_feature_impact
```

---

**Cell 2: Generowanie danych**
```python
print("Generowanie danych idealnych...")
df, feature_names = generate_ideal_data()
X = df[feature_names].values
y = df['Target'].values
```

**Odpowiedź**
```text
Generowanie danych idealnych...
```

---

**Cell 3: Podział i Normalizacja**
```python
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_train, X_test, scaler = normalize_data(X_train_raw, X_test_raw)
```

---

**Cell 4: Wizualizacja Danych**
```python
plot_distributions(df, feature_names, title_prefix='[Zad 1]')
plot_correlation_matrix(df, feature_names)
```

**Odpowiedź**

![](results/1.1.png)
![](results/1.2.png)


---

**Cell 5: Trening Modelu**
```python
print("Trenowanie modelu...")
model = train_model(X_train, y_train, C=1.0)
```

**Odpowiedź**
```text
Trenowanie modelu...
```

---

**Cell 6: Ewaluacja**
```python
metrics, y_pred, y_prob = evaluate_model(model, X_test, y_test)
print(f"Accuracy: {metrics['Accuracy']:.4f}")
print(f"F1 Score: {metrics['F1']:.4f}")
print(f"AUC:      {metrics['AUC']:.4f}")
```

**Odpowiedź**
```text
Accuracy: 1.0000
F1 Score: 1.0000
AUC:      1.0000
```

---

**Cell 7: Wizualizacja Wyników**
```python
plot_confusion_matrix(y_test, y_pred)
plot_roc_curve(metrics)
plot_betas(model, feature_names)

# Analiza wpływu cech dla 3 losowych próbek ze zbioru testowego
import numpy as np
indices = np.random.choice(len(X_test), 3, replace=False)
for idx in indices:
    prob = y_prob[idx]
    plot_feature_impact(model, X_test, feature_names, sample_idx=idx, prediction_prob=prob)
```

**Odpowiedź**

![](results/1.3.png)
![](results/1.4.png)
![](results/1.5.png)
![](results/1.6.png)
![](results/1.7.png)
![](results/1.8.png)


### Zadanie 2: EKSPERYMENT Z DANYMI REALISTYCZNYMI

Zadanie to symulowało rzeczywiste wyzwania w cyberbezpieczeństwie: silne niezbalansowanie klas (950 próbek normalnych vs 50 ataków) oraz zróżnicowany stopień trudności wykrycia zagrożeń. Ataki podzielono na oczywiste (przesunięcie o $4\sigma$), średnie ($2\sigma$) i subtelne ($1\sigma$), co wymusiło zastosowanie zaawansowanych strategii uczenia. Rozwiązanie porównywało trzy podejścia: standardową regresję logistyczną, model z ważeniem klas (`class_weight='balanced'`) oraz model z optymalizacją progu decyzyjnego $\tau_{opt}$. Kluczowym elementem była minimalizacja funkcji kosztu $C(\tau) = 100 \cdot FN + 1 \cdot FP$, kładącej nacisk na redukcję liczby niewykrytych ataków (False Negatives).

**Cell 1: Importy**
```python
import sys
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join('..', 'src')))
from data_generation import generate_realistic_data
from feature_engineering import normalize_data
from model_training import train_model, evaluate_model, optimize_threshold
from visualization import (plot_distributions, plot_confusion_matrix, plot_roc_curve, plot_task2_cost, plot_correlation_matrix, plot_betas, plot_feature_impact)
```

---

**Cell 2: Generowanie danych**
```python
print("Generowanie danych realistycznych (niezbalansowanych)...")
df, feature_names = generate_realistic_data()
X = df[feature_names].values
y = df['Target'].values
```

**Odpowiedź**
```text
Generowanie danych realistycznych (niezbalansowanych)...
```

---

**Cell 3: Podział i Normalizacja**
```python
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_train, X_test, scaler = normalize_data(X_train_raw, X_test_raw)
```

---

**Cell 4: Wizualizacja Danych**
```python
plot_distributions(df, feature_names, title_prefix='[Zad 2]')
plot_correlation_matrix(df, feature_names)
```

**Odpowiedź**

![](results/2.1.png)
![](results/2.2.png)


---

**Cell 5: Eksperyment - 3 Modele**
```python
# 1. Standard
print("Model Standard...")
model_std = train_model(X_train, y_train)
met_std, pred_std, prob_std = evaluate_model(model_std, X_test, y_test)

# 2. Balanced
print("Model Balanced...")
model_bal = train_model(X_train, y_train, class_weight='balanced')
met_bal, pred_bal, prob_bal = evaluate_model(model_bal, X_test, y_test)

# 3. Optimized Threshold (na modelu Standard)
print("Optymalizacja progu...")
tau_opt, costs, thresholds = optimize_threshold(model_std, X_test, y_test)
met_opt, pred_opt, _ = evaluate_model(model_std, X_test, y_test, threshold=tau_opt)
```

**Odpowiedź**
```text
Model Standard...
Model Balanced...
Optymalizacja progu...
```

---

**Cell 6: Wyniki**
```python
print(f"Standard FN: {met_std['FN']}")
print(f"Balanced FN: {met_bal['FN']}")
print(f"Optimized FN: {met_opt['FN']} (Tau={tau_opt:.2f})")
```

**Odpowiedź**
```text
Standard FN: 1
Balanced FN: 0
Optimized FN: 0 (Tau=0.12)
```

---

**Cell 7: Wizualizacja**
```python
# 1. Macierze pomyłek
plot_confusion_matrix(y_test, pred_std, title='Standard Model')
plot_confusion_matrix(y_test, pred_bal, title='Balanced Model')
plot_confusion_matrix(y_test, pred_opt, title=f'Optimized Threshold (Tau={tau_opt:.2f})')

# 2. Koszt
plot_task2_cost(thresholds, costs, tau_opt, min(costs))

# 3. Krzywe ROC (Porównanie Standard vs Balanced na jednym wykresie)
plt.figure(figsize=(8, 6))
plt.plot(met_std['FPR'], met_std['TPR'], label=f"Standard (AUC={met_std['AUC']:.3f})")
plt.plot(met_bal['FPR'], met_bal['TPR'], label=f"Balanced (AUC={met_bal['AUC']:.3f})", color='green')
plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

# 4. Porównanie Betas (Współczynników)
# Rysujemy obok siebie, aby zobaczyć różnicę w wagach
fig, ax = plt.subplots(1, 2, figsize=(16, 6))
beta_std = model_std.coef_[0]
beta_bal = model_bal.coef_[0]
idx = np.arange(len(feature_names))

ax[0].barh(feature_names, beta_std, color='blue')
ax[0].set_title("Betas: Standard Model")
ax[1].barh(feature_names, beta_bal, color='green')
ax[1].set_title("Betas: Balanced Model")
plt.tight_layout()
plt.show()

# 5. Analiza wpływu cech dla konkretnych typów ataków (Lokalna interpretowalność)
# Szukamy po jednej próbce z każdego rodzaju ataku w zbiorze testowym
print("\n--- Analiza wpływu cech dla różnych typów ataków (Balanced Model) ---")
# Musimy odzyskać typy ataków dla zbioru testowego 

# Wybieramy losowe próbki sklasyfikowane jako atak przez model Balanced
attack_indices = np.where(pred_bal == 1)[0]
if len(attack_indices) > 0:
    chosen_idx = attack_indices[0] # Pierwszy z brzegu wykryty atak
    prob = prob_bal[chosen_idx]
    print(f"Przykładowy wykryty atak (Index testowy: {chosen_idx})")
    plot_feature_impact(model_bal, X_test, feature_names, sample_idx=chosen_idx, prediction_prob=prob)
```

**Odpowiedź**

![](results/2.3.png)
![](results/2.4.png)
![](results/2.5.png)
![](results/2.6.png)
![](results/2.7.png)
![](results/2.8.png)

```text
--- Analiza wpływu cech dla różnych typów ataków (Balanced Model) ---
Przykładowy wykryty atak (Index testowy: 10)
```

![](results/2.9.png)

---

**Cell 8: Wykres liniowy pokazujący jak Precision(τ), Recall(τ) i F1(τ) zmieniają się w funkcji progu**
```python
# Nie ma tego w src, więc robimy to tutaj bezpośrednio
from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds_pr = precision_recall_curve(y_test, prob_std)
# F1 dla każdego progu
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)

plt.figure(figsize=(10, 6))
plt.plot(thresholds_pr, precisions[:-1], "b--", label="Precision")
plt.plot(thresholds_pr, recalls[:-1], "g-", label="Recall")
plt.plot(thresholds_pr, f1_scores[:-1], "r-", label="F1 Score")
plt.xlabel("Próg decyzyjny (Threshold)")
plt.ylabel("Wartość")
plt.title("Metryki vs Próg (Model Standard)")
plt.legend(loc="best")
plt.grid(True)
plt.show()
```

**Odpowiedź**

![](results/2.10.png)

### Zadanie 3: EKSPERYMENT Z DANYMI RZECZYWISTYMI (CICIDS2017)

Trzecie zadanie przeniosło problem detekcji na grunt danych rzeczywistych, wykorzystując zbiory CICIDS2017 lub UNSW-NB15. Głównym wyzwaniem była inżynieria cech (feature engineering) – konieczność przekształcenia surowych logów sieciowych w 7 cech bazowych zdefiniowanych w poprzednich zadaniach (np. obliczenie entropii portów czy stosunku SYN w oknach czasowych). Procedura wymagała zaawansowanego czyszczenia danych (usuwanie kolumn o zerowej wariancji, obsługa braków). Ostatecznie wytrenowano model na zbalansowanych wagach i porównano jego wyniki (Accuracy, Recall) z rezultatami uzyskanymi na danych syntetycznych, analizując przyczyny spadku skuteczności w realnym środowisku.

**Cell 1: Importy**
```python
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join('..', 'src')))
from data_generation import load_cicids_data
from feature_engineering import clean_cicids_data, engineer_features_cicids, normalize_data
from model_training import train_model, evaluate_model, analyze_errors_task3
from visualization import plot_distributions, plot_confusion_matrix, plot_roc_curve, plot_betas, plot_correlation_matrix, plot_learning_curve_manual
```

---

**Cell 2: Konfiguracja ścieżki**
```python
DATA_DIR = os.path.join('..', 'data', 'CICIDS2017')
```

---

**Cell 3: Wczytywanie danych**
```python
try:
    print("Wczytywanie surowych danych CICIDS2017...")
    df_raw = load_cicids_data(DATA_DIR, sample_size=50000)
    print(f"Wczytano {len(df_raw)} wierszy.")
except FileNotFoundError as e:
    print(e)
    print("Upewnij się, że pliki CSV znajdują się w folderze data/CICIDS2017/")
```

**Odpowiedź**
```text
Wczytywanie surowych danych CICIDS2017...
Wczytywanie: Monday-WorkingHours.pcap_ISCX.csv
Wczytywanie: Friday-WorkingHours-Afternoon-DDoS.pcap_ISCX.csv
Wczytywanie: Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
Wczytano 150000 wierszy.
```

---

**Cell 4: Przetwarzanie i Inżynieria Cech**
```python
print("Czyszczenie danych...")
df_clean = clean_cicids_data(df_raw)
print("Tworzenie cech...")
X_df, y, labels_raw = engineer_features_cicids(df_clean)
feature_names = X_df.columns.tolist()

# 1. Przygotowanie ramki do wizualizacji (łączenie cech z etykietą)
df_vis = X_df.copy()
df_vis['Target'] = y.values  # Dodajemy kolumnę Target, bo plot_distributions jej wymaga

# 2. Wyświetlenie rozkładów gęstości dla każdej cechy
print("Generowanie rozkładów gęstości...")
plot_distributions(df_vis, feature_names, title_prefix='[CICIDS2017]')

# 3. Wyświetlenie macierzy korelacji między cechami
print("Generowanie macierzy korelacji...")
plot_correlation_matrix(df_vis, feature_names)
```

**Odpowiedź**
```text
Czyszczenie danych...
Tworzenie cech...
Generowanie rozkładów gęstości...
```

[]![](results/3.1.png)
[]![](results/3.2.png)

---

**Cell 5: Podział na Train/Val/Test (60/20/20)**
```python
X_train_val, X_test_raw, y_train_val, y_test = train_test_split(X_df.values, y.values, test_size=0.2, stratify=y, random_state=42)
X_train_raw, X_val_raw, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, stratify=y_train_val, random_state=42)
```

---

**Cell 6: Normalizacja**
```python
X_train, X_test, X_val, scaler = normalize_data(X_train_raw, X_test_raw, X_val_raw)
```

---

**Cell 7: Trenowanie (Balanced)**
```python
print("Trenowanie modelu (Balanced)...")
model = train_model(X_train, y_train, class_weight='balanced')
```

**Odpowiedź**
```text
Trenowanie modelu (Balanced)...
```

---

**Cell 8: Ewaluacja**
```python
metrics, y_pred, y_prob = evaluate_model(model, X_test, y_test)
print(f"Accuracy:  {metrics['Accuracy']:.4f}")
print(f"Recall:    {metrics['Recall']:.4f}")
print(f"AUC:       {metrics['AUC']:.4f}")
```

**Odpowiedź**
```text
Accuracy:  0.8534
Recall:    0.9840
AUC:       0.8918
```

---

**Cell 9: Wizualizacja**
```python
plot_confusion_matrix(y_test, y_pred)
plot_roc_curve(metrics)
plot_betas(model, feature_names)
```

**Odpowiedź**

![](results/3.3.png)
![](results/3.4.png)
![](results/3.5.png)

---

**Cell 9b: Generowanie i rysowanie krzywej uczenia**
```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

train_sizes_pct = [0.1, 0.25, 0.5, 0.75, 1.0]
train_scores = []
val_scores = []
train_sizes_abs = []

print("Generowanie krzywej uczenia...")
for pct in train_sizes_pct:
    # Bierzemy podzbiór danych treningowych
    n_samples = int(pct * len(X_train))
    train_sizes_abs.append(n_samples)
    
    X_sub = X_train[:n_samples]
    y_sub = y_train[:n_samples]
    
    # Trenujemy tymczasowy model
    model_temp = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    model_temp.fit(X_sub, y_sub)
    
    # Zapisujemy wyniki
    train_scores.append(accuracy_score(y_sub, model_temp.predict(X_sub)))
    val_scores.append(accuracy_score(y_val, model_temp.predict(X_val)))

plot_learning_curve_manual(train_sizes_abs, train_scores, val_scores)
```

**Odpowiedź**
```text
Generowanie krzywej uczenia...
```

![](results/3.6.png)

---

**Cell 10: Analiza Błędów**
```python
fn_samples, fp_samples, _, _ = analyze_errors_task3(X_test_raw, y_test, y_pred, y_prob, feature_names)

print("\nPrzykładowe False Negatives (Atak uznany za normę):")
display(fn_samples.head())

print("\nPrzykładowe False Positives (Norma uznana za atak):")
display(fp_samples.head())
```

**Odpowiedź**

**Przykładowe False Negatives (Atak uznany za normę)**
|     | packetspersec | avgpacketsize | portentropy | synratio | uniquedstips | connectionduration | repeatedconnections |
| --- | ------------- | ------------- | ----------- | -------- | ------------ | ------------------ | ------------------- |
| 68  | 250000.000000 | 3.000000      | 20.0        | 0.0      | 6.0          | 0.000008           | 0.0                 |
| 345 | 200000.000000 | 5.000000      | 20.0        | 0.0      | 6.0          | 0.000010           | 0.0                 |
| 366 | 19.655075     | 1290.333333   | 200.0       | 0.0      | 11595.0      | 0.457897           | 456311.0            |
| 475 | 250000.000000 | 5.000000      | 20.0        | 0.0      | 6.0          | 0.000008           | 0.0                 |
| 584 | 333333.333300 | 3.000000      | 20.0        | 0.0      | 6.0          | 0.000006           | 0.0                 |

**Przykładowe False Positives (Norma uznana za atak)**
|     | packetspersec | avgpacketsize | portentropy | synratio | uniquedstips | connectionduration | repeatedconnections |
| --- | ------------- | ------------- | ----------- | -------- | ------------ | ------------------ | ------------------- |
| 0   | 86956.521739  | 9.0           | 20.0        | 0.0      | 6.0          | 0.000023           | 0.0                 |
| 2   | 18691.588790  | 9.0           | 20.0        | 0.0      | 6.0          | 0.000107           | 0.0                 |
| 8   | 42553.191489  | 9.0           | 20.0        | 0.0      | 6.0          | 0.000047           | 0.0                 |
| 19  | 26315.789474  | 9.0           | 20.0        | 0.0      | 6.0          | 0.000076           | 0.0                 |
| 24  | 666666.666667 | 9.0           | 0.0         | 0.0      | 0.0          | 0.000003           | 3.0                 |