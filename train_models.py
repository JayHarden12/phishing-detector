import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.inspection import permutation_importance
import joblib
import json
import os

# Configuration
DATA_PATH = "data/Phishing_Legitimate_full.csv"
MODEL_DIR = "ml_models"

def compute_risk_coverage(y_true, y_prob, thresholds=np.linspace(0.5, 0.99, 50)):
    curve = []
    for t in thresholds:
        upper = t
        lower = 1 - t
        # Accepted are those where prob > upper or prob < lower
        # We predict 1 if prob > upper, 0 if prob < lower
        accepted_idx = (y_prob >= upper) | (y_prob <= lower)
        n_accepted = np.sum(accepted_idx)
        coverage = n_accepted / len(y_true)
        
        if n_accepted > 0:
            y_pred_accepted = (y_prob[accepted_idx] >= upper).astype(int)
            y_true_accepted = y_true[accepted_idx]
            selection_risk = 1.0 - accuracy_score(y_true_accepted, y_pred_accepted)
        else:
            selection_risk = 0.0
            
        curve.append({
            "threshold": float(upper),
            "coverage": float(coverage),
            "risk": float(selection_risk),
            "abstention_rate": float(1 - coverage)
        })
    return curve

def main():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    
    # Check what CLASS_LABEL unique values are
    print("Unique labels:", df['CLASS_LABEL'].unique())
    # Typically 1 = Phishing, 0 = Legitimate or something like that. 
    # Let's map so that 1 = Phishing, 0 = Legitimate for standard setup.
    # In some datasets, 1 is Legitimate, 0 is Phishing or 1 and -1. Let's see unique values.
    
    # Assuming 'id' is a column, we drop it
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
        
    X = df.drop(columns=['CLASS_LABEL'])
    y = df['CLASS_LABEL']
    
    # If the labels are 0 and 1, we assume 1 is Phishing for the sake of the system
    # If they are 1 and -1, we map -1 to 0
    if -1 in y.values:
        y = y.replace(-1, 0)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.joblib'))
    
    models = {
        'naive_bayes': GaussianNB(),
        'svm': SVC(probability=True, random_state=42),
        'logistic_regression': LogisticRegression(max_iter=1000, random_state=42)
    }
    
    metrics = {"classifiers": []}
    
    print("Training models...")
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_scaled, y_train)
        joblib.dump(model, os.path.join(MODEL_DIR, f"{name}.joblib"))
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1] # Probability of class 1
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred).tolist()
        
        # Risk-coverage curve
        curve = compute_risk_coverage(y_test.values, y_prob)
        
        # Feature importance (if applicable)
        importance = None
        if name == 'logistic_regression':
            importance = dict(zip(X.columns, model.coef_[0].tolist()))
        else:
            # Calculate permutation importance for non-linear models
            print(f"Calculating permutation importance for {name}...")
            # We use a smaller sample to speed up calculation, default 5 repeats
            r = permutation_importance(model, X_test_scaled, y_test, n_repeats=5, random_state=42, n_jobs=-1)
            importance = dict(zip(X.columns, r.importances_mean.tolist()))
        
        metrics["classifiers"].append({
            "name": name,
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
            "roc_auc": float(roc_auc),
            "confusion_matrix": cm,
            "risk_coverage_curve": curve,
            "feature_importance": importance
        })
        print(f"{name} - Acc: {acc:.4f}, AUC: {roc_auc:.4f}")
        
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
        
    # Save a sample of X_test for testing if needed
    pd.DataFrame(X_test_scaled, columns=X.columns).to_csv(os.path.join(MODEL_DIR, "sample_features.csv"), index=False)
        
    print("Training complete and models saved.")

if __name__ == "__main__":
    main()
