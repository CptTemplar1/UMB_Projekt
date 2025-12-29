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

### Zadanie 2: EKSPERYMENT Z DANYMI REALISTYCZNYMI

### Zadanie 3: EKSPERYMENT Z DANYMI RZECZYWISTYMI (CICIDS2017)