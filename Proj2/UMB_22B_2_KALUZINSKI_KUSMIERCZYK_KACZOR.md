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

### Zadanie 1: EKSPERYMENT Z DANYMI IDEALNYMI

Wygeneruj syntetyczny zbiór danych symulujący ruch sieciowy składający się z 1000 próbek, gdzie 800 próbek reprezentuje normalny ruch a 200 próbek reprezentuje ataki. Każda próbka musi zawierać 7 cech numerycznych charakteryzujących ruch sieciowy: liczbę pakietów na sekundę, średni rozmiar pakietu w bajtach, entropię portów docelowych, stosunek pakietów SYN do wszystkich pakietów, liczbę unikalnych adresów IP docelowych, czas trwania połączenia w sekundach oraz liczbę powtórzonych połączeń.

Normalny ruch wygeneruj losując wartości z rozkładów o następujących parametrach:
* liczba pakietów na sekundę: N(μ = 50, σ = 15)
* średni rozmiar pakietu: N(μ = 800, σ = 200)
* entropia portów: N(μ = 2.5, σ = 0.5)
* stosunek SYN: N(μ = 0.2, σ = 0.05), obcięte do przedziału [0, 1]
* liczba unikalnych IP: Poisson(λ = 5) + 1
* czas trwania połączenia: Exp(λ = 1/30)
* liczba powtórzeń: Poisson(λ = 2)

Ataki wygeneruj używając tych samych typów rozkładów ale z przesunięcymi parametrami tak aby były wyraźnie oddzielone od normalnego ruchu:
* liczba pakietów na sekundę: N(μ = 250, σ = 30)
* średni rozmiar pakietu: N(μ = 300, σ = 100)
* entropia portów: N(μ = 4.0, σ = 0.3)
* stosunek SYN: N(μ = 0.8, σ = 0.05), obcięte do przedziału [0, 1]
* liczba unikalnych IP: Poisson(λ = 50) + 1
* czas trwania połączenia: Exp(λ = 1/2)
* liczba powtórzeń: Poisson(λ = 20)

Podziel wygenerowane dane na zbiór treningowy zawierający 70% próbek oraz zbiór testowy zawierający 30% próbek zachowując proporcje klas w obu zbiorach. Znormalizuj wszystkie cechy obliczając dla zbioru treningowego średnią $\hat{\mu}$ i odchylenie standardowe $\hat{\sigma}$ każdej cechy, a następnie stosując transformację $z_{train} = \frac{x_{train} - \hat{\mu}}{\hat{\sigma}}$ oraz $z_{test} = \frac{x_{test} - \hat{\mu}}{\hat{\sigma}}$.

Wytrenuj model regresji logistycznej z regularyzacją L2 o sile regularyzacji C = 1.0. Oblicz prognozy na zbiorze testowym i wyznacz macierz pomyłek, dokładność Accuracy = (TP+TN)/(TP+TN+FP+FN), precyzję Precision = TP/(TP +FP), czułość Recall = TP/(TP +FN) oraz miarę F1 F1 = 2· Precision · Recall/(Precision+Recall) osobno dla obu klas. Oblicz pole pod krzywą ROC oznaczone jako AUC. Wyodrębnij współczynniki $\beta_1, \beta_2, \dots, \beta_7$ dla każdej cechy oraz wyraz wolny $/beta_0$ z wytrenowanego modelu.

Wygeneruj następujące wizualizacje: 4 wykresy pokazujące rozkłady gęstości dla liczby pakietów na sekundę, średniego rozmiaru pakietu, entropii portów oraz stosunku SYN, gdzie każdy wykres zawiera 2 krzywe przedstawiające rozkład dla normalnego ruchu i dla ataków w różnych kolorach; wykres słupkowy poziomy przedstawiający współczynniki $\beta_1, \dots, \beta_7$ uporządkowane według wartości $|\beta_i|$ z adnotacjami liczbowymi; mapę ciepła macierzy pomyłek z liczbami w komórkach; krzywą ROC z zaznaczonym punktem optymalnym i wartością AUC w legendzie; histogram prawdopodobieństw predykcji P(y = 1|x) osobno dla próbek normalnych i ataków z linią pionową przy progu τ = 0.5; 3 wykresy słupkowe pokazujące wpływ każdej cechy $|\beta_i| · x_i$ na decyzję modelu dla 3 wybranych próbek testowych; macierz korelacji Pearsona ρij między wszystkimi cechami jako mapę ciepła.

Przeanalizuj wyniki odpowiadając na pytania: które cechy mają największe współczynniki $|\beta_i|$ i co to oznacza dla detekcji ataków, ile błędów typu FN i FP popełnia model, czy wszystkie 7 cech jest potrzebnych do osiągnięcia wysokiej dokładności, oraz które pary cech mają najwyższe wartości |ρij |.

**Procedura postępowania**  
Szczegółowa procedura postępowania dla tego zadania została opisana w Algorytmie 1.