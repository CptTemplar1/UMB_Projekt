import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

def train_model(X_train, y_train, class_weight=None, C=1.0, max_iter=1000):
    """Trenuje model regresji logistycznej."""
    model = LogisticRegression(penalty='l2', C=C, class_weight=class_weight, 
                               solver='lbfgs', max_iter=max_iter, random_state=42)
    model.fit(X_train, y_train)
    return model

def optimize_threshold(model, X_test, y_test, wFN=100, wFP=1):
    """Znajduje optymalny próg decyzyjny (Zadanie 2)."""
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

def evaluate_model(model, X_test, y_test, threshold=0.5):
    """Oblicza metryki dla modelu."""
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

def analyze_errors_task3(X_test_raw, y_test, y_pred, y_prob, feature_names):
    """Analiza FN/FP dla Zadania 3."""
    X_df = pd.DataFrame(X_test_raw, columns=feature_names)
    
    # Indeksy błędów
    fn_idx = np.where((y_test == 1) & (y_pred == 0))[0]
    fp_idx = np.where((y_test == 0) & (y_pred == 1))[0]
    
    return X_df.iloc[fn_idx], X_df.iloc[fp_idx], fn_idx, fp_idx