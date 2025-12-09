import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def clean_cicids_data(df):
    """Realizuje Algorytm 5 (Czyszczenie)."""
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

def engineer_features_cicids(df):
    """Realizuje Algorytm 6 (Tworzenie 7 cech bazowych)."""
    # Uproszczona inżynieria cech oparta na dostępnych kolumnach (bez ciężkiej agregacji czasowej)
    # aby kod działał w rozsądnym czasie na standardowym komputerze.
    
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
                     
    # Sprzątanie finalne (na wszelki wypadek)
    features.replace([np.inf, -np.inf], 0, inplace=True)
    features.fillna(0, inplace=True)
    
    return features[feature_names], y, df['Label'] # Zwracamy też oryginalne etykiety tekstowe

def normalize_data(X_train, X_test, X_val=None):
    """Normalizacja (StandardScaler) fit na train, transform na resztę."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    if X_val is not None:
        X_val_scaled = scaler.transform(X_val)
        return X_train_scaled, X_test_scaled, X_val_scaled, scaler
        
    return X_train_scaled, X_test_scaled, scaler