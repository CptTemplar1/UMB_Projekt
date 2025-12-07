import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

# Konfiguracja stylu wykresów
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

def generate_data(n_norm=800, n_attack=200, seed=42):
    """
    Generuje syntetyczny zbiór danych zgodnie z opisem zadania.
    """
    np.random.seed(seed)
    
    # Nazwy cech
    feature_names = [
        'Liczba pakietów/s', 
        'Średni rozmiar pakietu', 
        'Entropia portów', 
        'Stosunek SYN', 
        'Liczba unikalnych IP', 
        'Czas trwania', 
        'Liczba powtórzeń'
    ]
    
    # --- Generowanie Ruchu Normalnego (0) ---
    X_norm = np.zeros((n_norm, 7))
    X_norm[:, 0] = np.random.normal(50, 15, n_norm)              # Pakiety/s
    X_norm[:, 1] = np.random.normal(800, 200, n_norm)            # Rozmiar pakietu
    X_norm[:, 2] = np.random.normal(2.5, 0.5, n_norm)            # Entropia
    X_norm[:, 3] = np.clip(np.random.normal(0.2, 0.05, n_norm), 0, 1) # Stosunek SYN
    X_norm[:, 4] = np.random.poisson(5, n_norm) + 1              # Unikalne IP
    X_norm[:, 5] = np.random.exponential(30, n_norm)             # Czas (scale=1/lambda -> 1/(1/30)=30)
    X_norm[:, 6] = np.random.poisson(2, n_norm)                  # Powtórzenia
    y_norm = np.zeros(n_norm)

    # --- Generowanie Ataków (1) ---
    X_attack = np.zeros((n_attack, 7))
    X_attack[:, 0] = np.random.normal(250, 30, n_attack)
    X_attack[:, 1] = np.random.normal(300, 100, n_attack)
    X_attack[:, 2] = np.random.normal(4.0, 0.3, n_attack)
    X_attack[:, 3] = np.clip(np.random.normal(0.8, 0.05, n_attack), 0, 1)
    X_attack[:, 4] = np.random.poisson(50, n_attack) + 1
    X_attack[:, 5] = np.random.exponential(2, n_attack)          # Czas (scale=1/(1/2)=2)
    X_attack[:, 6] = np.random.poisson(20, n_attack)
    y_attack = np.ones(n_attack)

    # Łączenie danych
    X = np.vstack([X_norm, X_attack])
    y = np.hstack([y_norm, y_attack])
    
    # Tworzenie DataFrame dla łatwiejszej obsługi przed podziałem
    df = pd.DataFrame(X, columns=feature_names)
    df['Target'] = y
    
    return df, feature_names

def run_experiment():
    print(">>> Rozpoczynanie Zadania 1: Eksperyment z danymi idealnymi...")
    
    # 1. Generowanie danych
    df, feature_names = generate_data()
    X = df[feature_names].values
    y = df['Target'].values

    # 2. Podział i Normalizacja (Algorytm 1: linie 27-35)
    # Stratified split 70/30
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42, shuffle=True
    )
    
    # Normalizacja (fit na train, transform na train i test)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    
    # 3. Modelowanie (Algorytm 1: linie 36-40)
    # Regresja Logistyczna z L2 (C=1.0)
    model = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', random_state=42)
    model.fit(X_train, y_train)
    
    # Pobranie współczynników
    beta = model.coef_[0]
    beta0 = model.intercept_[0]
    
    # 4. Predykcja i Ewaluacja (Algorytm 1: linie 42-53)
    # Prawdopodobieństwa
    y_prob = model.predict_proba(X_test)[:, 1]
    # Klasy (próg 0.5)
    y_pred = (y_prob >= 0.5).astype(int)
    
    # Metryki
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    accuracy = accuracy_score(y_test, y_pred)
    precision_norm = precision_score(y_test, y_pred, pos_label=0) # dla klasy 0
    recall_norm = recall_score(y_test, y_pred, pos_label=0)
    f1_norm = f1_score(y_test, y_pred, pos_label=0)
    
    precision_attack = precision_score(y_test, y_pred, pos_label=1) # dla klasy 1
    recall_attack = recall_score(y_test, y_pred, pos_label=1)
    f1_attack = f1_score(y_test, y_pred, pos_label=1)
    
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    print("\n--- WYNIKI MODELU ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC:      {roc_auc:.4f}")
    print(f"Macierz pomyłek: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    print("-" * 30)
    print(f"Klasa Normal (0): Pre={precision_norm:.3f}, Rec={recall_norm:.3f}, F1={f1_norm:.3f}")
    print(f"Klasa Atak (1):   Pre={precision_attack:.3f}, Rec={recall_attack:.3f}, F1={f1_attack:.3f}")
    
    # 5. Wizualizacje
    create_visualizations(df, feature_names, beta, beta0, y_test, y_prob, y_pred, fpr, tpr, roc_auc, X_test, model)
    
    # 6. Analiza (odpowiedzi na pytania)
    perform_analysis(beta, feature_names, fn, fp, df)

def create_visualizations(df, feature_names, beta, beta0, y_test, y_prob, y_pred, fpr, tpr, roc_auc, X_test, model):
    fig = plt.figure(figsize=(20, 25))
    
    # A. 4 Wykresy gęstości (KDE)
    features_to_plot = ['Liczba pakietów/s', 'Średni rozmiar pakietu', 'Entropia portów', 'Stosunek SYN']
    indices = [0, 1, 2, 3] # Indeksy tych cech w feature_names
    
    for i, fname in enumerate(features_to_plot):
        ax = fig.add_subplot(5, 2, i+1)
        sns.kdeplot(data=df[df['Target']==0], x=fname, fill=True, color='blue', label='Normalny', ax=ax)
        sns.kdeplot(data=df[df['Target']==1], x=fname, fill=True, color='red', label='Atak', ax=ax)
        ax.set_title(f'Rozkład gęstości: {fname}')
        ax.legend()

    # B. Wykres słupkowy współczynników beta
    ax_beta = fig.add_subplot(5, 2, 5)
    coef_df = pd.DataFrame({'Cecha': feature_names, 'Beta': beta, 'AbsBeta': np.abs(beta)})
    coef_df = coef_df.sort_values('AbsBeta', ascending=True)
    
    bars = ax_beta.barh(coef_df['Cecha'], coef_df['Beta'], color=np.where(coef_df['Beta']>0, 'crimson', 'royalblue'))
    ax_beta.set_title('Współczynniki regresji logicznej (posortowane wg |Beta|)')
    ax_beta.bar_label(bars, fmt='%.2f')

    # C. Mapa ciepła macierzy pomyłek
    ax_conf = fig.add_subplot(5, 2, 6)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_conf, 
                xticklabels=['Pred: Normal', 'Pred: Atak'], yticklabels=['Real: Normal', 'Real: Atak'])
    ax_conf.set_title('Macierz Pomyłek')

    # D. Krzywa ROC
    ax_roc = fig.add_subplot(5, 2, 7)
    ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
    ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    # Punkt optymalny (najbliższy (0,1))
    optimal_idx = np.argmax(tpr - fpr)
    ax_roc.scatter(fpr[optimal_idx], tpr[optimal_idx], c='red', s=100, label='Punkt optymalny')
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('Krzywa ROC')
    ax_roc.legend(loc="lower right")

    # E. Histogram prawdopodobieństw
    ax_hist = fig.add_subplot(5, 2, 8)
    sns.histplot(y_prob[y_test==0], color='blue', label='Normalny', kde=False, bins=20, alpha=0.6, ax=ax_hist)
    sns.histplot(y_prob[y_test==1], color='red', label='Atak', kde=False, bins=20, alpha=0.6, ax=ax_hist)
    ax_hist.axvline(0.5, color='black', linestyle='--', label='Próg (0.5)')
    ax_hist.set_title('Histogram prawdopodobieństw P(y=1|x)')
    ax_hist.set_xlabel('Prawdopodobieństwo ataku')
    ax_hist.legend()

    # F. Wpływ cech dla 3 próbek (lokalna interpretowalność: beta_i * x_i)
    # Wybieramy 3 losowe próbki ze zbioru testowego
    np.random.seed(101)
    sample_indices = np.random.choice(len(X_test), 3, replace=False)
    
    for i, idx in enumerate(sample_indices):
        ax_local = fig.add_subplot(5, 3, 13+i) # Wiersz 5, kolumny 1-3
        sample_x = X_test[idx]
        impact = beta * sample_x # Wektor wpływu cech
        
        # Sortowanie dla czytelności
        impact_df = pd.DataFrame({'Cecha': feature_names, 'Wplyw': impact})
        impact_df = impact_df.sort_values('Wplyw')
        
        colors = ['red' if x > 0 else 'blue' for x in impact_df['Wplyw']]
        ax_local.barh(impact_df['Cecha'], impact_df['Wplyw'], color=colors)
        true_label = "Atak" if y_test[idx] == 1 else "Normalny"
        pred_prob = y_prob[idx]
        ax_local.set_title(f'Próbka #{idx} (Rzeczyw: {true_label})\nP(Atak)={pred_prob:.2f}')
        ax_local.set_xlabel('Wpływ na decyzję (Log-odds)')

    # G. Macierz korelacji
    plt.tight_layout()
    plt.show()
    
    # Osobne okno/figura dla korelacji, by było czytelne
    plt.figure(figsize=(10, 8))
    corr_matrix = df[feature_names].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    plt.title('Macierz korelacji Pearsona cech')
    plt.show()

def perform_analysis(beta, feature_names, fn, fp, df):
    print("\n" + "="*50)
    print("ANALIZA WYNIKÓW")
    print("="*50)
    
    # 1. Analiza współczynników
    coef_dict = dict(zip(feature_names, beta))
    sorted_coefs = sorted(coef_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    
    print("\n1. Które cechy mają największe współczynniki |beta_i|?")
    print("-" * 40)
    for name, val in sorted_coefs[:3]:
        print(f"   - {name}: {val:.4f}")
    
    print("\n   Co to oznacza?")
    print("   Cechy o wysokich dodatnich wartościach beta (np. Stosunek SYN, Liczba pakietów/s) silnie wskazują na klasę 'Atak'.")
    print("   Cechy o ujemnych wartościach beta (np. Czas trwania) silnie wskazują na klasę 'Normalny' (krótkie czasy są typowe dla tego ataku w danych, ale długie dla normalnego).")
    print("   Z uwagi na idealną separację danych, model przypisuje duże wagi cechom najlepiej rozdzielającym rozkłady.")

    # 2. Analiza błędów
    print(f"\n2. Ile błędów typu FN i FP popełnia model?")
    print("-" * 40)
    print(f"   - Fałszywie Pozytywne (FP): {fp} (Normalny uznany za Atak)")
    print(f"   - Fałszywie Negatywne (FN): {fn} (Atak uznany za Normalny)")
    if fp == 0 and fn == 0:
        print("   Wniosek: Dane są 'idealne', rozkłady praktycznie na siebie nie zachodzą, co pozwala na perfekcyjną separację liniową.")

    # 3. Redundancja cech
    print("\n3. Czy wszystkie 7 cech jest potrzebnych?")
    print("-" * 40)
    print("   Patrząc na wykresy gęstości, 'Liczba pakietów/s' oraz 'Stosunek SYN' same w sobie oferują niemal idealną separację.")
    print("   Cechy o bardzo małych współczynnikach beta wnoszą niewielką wartość informacyjną przy obecności silniejszych cech.")
    
    # 4. Korelacja
    corr_matrix = df[feature_names].corr().abs()
    # Maskujemy przekątną
    np.fill_diagonal(corr_matrix.values, 0)
    max_corr = corr_matrix.unstack().sort_values(ascending=False).drop_duplicates().head(3)
    
    print("\n4. Które pary cech mają najwyższe wartości |rho_ij|?")
    print("-" * 40)
    for idx, val in max_corr.items():
        print(f"   - {idx[0]} <-> {idx[1]}: {val:.3f}")
    print("   Wysoka korelacja wynika ze struktury generowania danych: klasa 'Atak' ma wysokie wartości w wielu cechach jednocześnie,")
    print("   tworząc naturalną korelację w całym zbiorze danych, mimo że cechy generowane były niezależnie.")

if __name__ == "__main__":
    run_experiment()