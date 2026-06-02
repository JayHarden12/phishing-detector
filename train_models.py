import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.inspection import permutation_importance

class SoftVotingEnsemble:
    def __init__(self, models):
        self.models = models
        
    def predict_proba(self, X):
        probas = [m.predict_proba(X) for m in self.models.values()]
        return np.mean(probas, axis=0)
        
    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)

# Create directories
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

print("Loading dataset...")
df = pd.read_csv('data/Phishing_Legitimate_full.csv')
X = df.drop(['id', 'CLASS_LABEL'], axis=1)
y = df['CLASS_LABEL']
feature_names = X.columns.tolist()

# 60/10/30 split
print("Splitting dataset (60% Train, 10% Val, 30% Test)...")
X_train_temp, X_test, y_train_temp, y_test = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
# We have 70% in temp. We need 60% train and 10% val overall.
# So val size within temp is 10/70 = 1/7
X_train, X_val, y_train, y_val = train_test_split(X_train_temp, y_train_temp, test_size=(1/7), stratify=y_train_temp, random_state=42)

print(f"Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")

print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, 'models/scaler.joblib')
joblib.dump(feature_names, 'models/feature_names.joblib')

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models_config = {
    'Naive_Bayes': {
        'estimator': GaussianNB(),
        'param_grid': {
            'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
        }
    },
    'Logistic_Regression': {
        'estimator': LogisticRegression(max_iter=2000, random_state=42),
        'param_grid': {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear']
        }
    },
    'SVM': {
        'estimator': SVC(probability=True, random_state=42),
        'param_grid': {
            'C': [1, 10],
            'gamma': ['scale'],
            'kernel': ['rbf']
        }
    }
}

best_models = {}
evaluation_summary = {}

# 1. Train models and evaluate
for name, config in models_config.items():
    print(f"\nTraining {name}...")
    grid = GridSearchCV(
        estimator=config['estimator'],
        param_grid=config['param_grid'],
        scoring='f1',
        cv=cv,
        n_jobs=-1,
        verbose=1
    )
    grid.fit(X_train_scaled, y_train)
    
    best_model = grid.best_estimator_
    best_models[name] = best_model
    joblib.dump(best_model, f'models/{name.lower()}.joblib')
    
    print(f"Best params for {name}: {grid.best_params_}")
    
    # Evaluate on test set
    y_pred = best_model.predict(X_test_scaled)
    y_prob = best_model.predict_proba(X_test_scaled)[:, 1] # Probability of Legitimate (1)
    
    # Store metrics
    evaluation_summary[name] = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred)),
        'recall': float(recall_score(y_test, y_pred)),
        'f1': float(f1_score(y_test, y_pred)),
        'roc_auc': float(roc_auc_score(y_test, y_prob))
    }
    
    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Phishing (0)', 'Legitimate (1)'], 
                yticklabels=['Phishing (0)', 'Legitimate (1)'])
    plt.title(f'{name} Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(f'results/confusion_matrix_{name}.png', dpi=300)
    plt.close()

# Evaluate Ensemble
print("\nEvaluating Ensemble...")
ensemble_model = SoftVotingEnsemble(best_models.copy())
best_models['Ensemble'] = ensemble_model # Add to dict so ROC/Risk-Coverage loops pick it up

y_pred_ens = ensemble_model.predict(X_test_scaled)
y_prob_ens = ensemble_model.predict_proba(X_test_scaled)[:, 1]

evaluation_summary['Ensemble'] = {
    'accuracy': float(accuracy_score(y_test, y_pred_ens)),
    'precision': float(precision_score(y_test, y_pred_ens)),
    'recall': float(recall_score(y_test, y_pred_ens)),
    'f1': float(f1_score(y_test, y_pred_ens)),
    'roc_auc': float(roc_auc_score(y_test, y_prob_ens))
}

cm_ens = confusion_matrix(y_test, y_pred_ens)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_ens, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Phishing (0)', 'Legitimate (1)'], 
            yticklabels=['Phishing (0)', 'Legitimate (1)'])
plt.title('Ensemble Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('results/confusion_matrix_Ensemble.png', dpi=300)
plt.close()

# Save metrics
with open('results/evaluation_summary.json', 'w') as f:
    json.dump(evaluation_summary, f, indent=4)

print("\nSaved evaluation summary.")

# 2. ROC Curves
print("Generating ROC curves...")
plt.figure(figsize=(8, 6))
for name, model in best_models.items():
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curves')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('results/roc_curves.png', dpi=300)
plt.close()

# 3. Selective Rejection on Validation Set
print("Generating Risk-Coverage trade-off...")
plt.figure(figsize=(10, 6))

thresholds = np.arange(0.50, 1.00, 0.05)
for name, model in best_models.items():
    y_val_prob = model.predict_proba(X_val_scaled)[:, 1]
    
    coverages = []
    risks = []
    
    for tu in thresholds:
        tl = 1.0 - tu
        
        # P(Legitimate) > tu -> Predict Legitimate (1)
        # P(Legitimate) < tl -> Predict Phishing (0)
        # Else -> Abstain
        
        accepted_mask = (y_val_prob > tu) | (y_val_prob < tl)
        accepted_count = np.sum(accepted_mask)
        coverage = accepted_count / len(y_val)
        
        if accepted_count > 0:
            y_val_accepted_true = y_val[accepted_mask]
            y_val_accepted_pred = (y_val_prob[accepted_mask] > tu).astype(int)
            error_rate = 1.0 - accuracy_score(y_val_accepted_true, y_val_accepted_pred)
        else:
            error_rate = 0.0
            
        coverages.append(coverage)
        risks.append(error_rate)
        
    plt.plot(coverages, risks, marker='o', label=name)

plt.xlabel('Coverage (Proportion of URLs classified)')
plt.ylabel('Selection Risk (Error rate on accepted predictions)')
plt.title('Risk-Coverage Trade-off Curve (Validation Set)')
plt.legend()
plt.grid(alpha=0.3)
plt.gca().invert_xaxis() # High coverage on left, low coverage on right
plt.tight_layout()
plt.savefig('results/risk_coverage.png', dpi=300)
plt.close()

# 4. Feature Importance (using Random Forest or Permutation Importance on Best Model)
print("Calculating feature importance (Permutation Importance on Logistic Regression)...")
# We use Logistic Regression for simplicity and speed of permutation importance
if 'Logistic_Regression' in best_models:
    lr_model = best_models['Logistic_Regression']
    perm_importance = permutation_importance(lr_model, X_test_scaled, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    
    sorted_idx = perm_importance.importances_mean.argsort()[-20:] # Top 20
    
    plt.figure(figsize=(10, 8))
    plt.boxplot(
        perm_importance.importances[sorted_idx].T,
        vert=False,
        labels=np.array(feature_names)[sorted_idx],
    )
    plt.title("Top 20 Features (Permutation Importance - Logistic Regression)")
    plt.xlabel("Decrease in accuracy score")
    plt.tight_layout()
    plt.savefig('results/feature_importance.png', dpi=300)
    plt.close()

print("All tasks completed successfully. Artifacts saved in models/ and results/.")
