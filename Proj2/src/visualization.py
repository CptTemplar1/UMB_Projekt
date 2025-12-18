import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import learning_curve

# Rysuje rozkłady gęstości dla cech
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

# Rysuje macierz korelacji Pearsona
def plot_correlation_matrix(df, feature_names):
    plt.figure(figsize=(10, 8))
    # Obliczamy korelację tylko dla cech numerycznych
    corr = df[feature_names].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool)) # Opcjonalnie: maskowanie górnego trójkąta
    
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    plt.title("Macierz korelacji Pearsona")
    plt.show()

# Rysuje wykres słupkowy współczynników modelu
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

# Rysuje wpływ cech na decyzję dla KONKRETNEJ próbki (lokalna interpretowalność)
# Wykres pokazuje beta_i * x_i.
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

# Rysuje macierz pomyłek z dodatkowymi informacjami takimi jak procenty i liczby
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

# Rysuje krzywą ROC 
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

# Rysuje funkcję kosztu dla Zadania 2
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

# Rysuje krzywą uczenia dla Zadania 3
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