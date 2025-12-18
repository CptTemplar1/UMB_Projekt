import numpy as np
import pandas as pd
import os

# Generowanie danych do Zadania 1 (Idealne/Zbalansowane)
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

# Generowanie danych do Zadania 2 (Realistyczne/Niezbalansowane)
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

# Wczytywanie danych CICIDS 2017 z plików CSV dla Zadania 3
# data_dir: ścieżka do folderu data/CICIDS2017/
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