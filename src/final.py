import os
import joblib
import numpy as np
import pandas as pd
import datetime
from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

def preprocess_data(df, threshold, random_state):
    y = df.iloc[:, 0].map({'non egg': 0, 'egg': 1})
    X = df.iloc[:, 1:]
    all_features = X.columns.tolist()
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=random_state)
    corr_matrix = X_train.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), 1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    X_train_filt = X_train.drop(columns=to_drop)
    X_test_filt = X_test.drop(columns=to_drop)
    X_filt = X.drop(columns=to_drop)
    features_after_corr = X_train_filt.columns.tolist()
    return X_filt, X_train_filt, X_test_filt, y_train, y_test, all_features, to_drop, features_after_corr

def main():
    model_path = "outputs/final_model.pkl"
    data_path = "data/data-og.xlsx"
    sheet = "dtd_sv_xtrafeat"
    threshold = 0.65
    random_state = 42
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs("outputs", exist_ok=True)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at: {model_path}")
    model = joblib.load(model_path)
    df = pd.read_excel(data_path, engine="openpyxl", sheet_name=sheet)
    X, X_train, X_test, y_train, y_test, all_features, dropped_corr, features_after_corr = preprocess_data(df, threshold, random_state)
    kbest = model.named_steps.get("kbest", None)
    if kbest:
        support_mask = kbest.get_support()
        selected_kbest = list(X_train.columns[support_mask])
        kbest_scores = kbest.scores_[support_mask]
        kbest_pvalues = kbest.pvalues_[support_mask]
    else:
        selected_kbest = features_after_corr
        kbest_scores = None
        kbest_pvalues = None
    final_features_used = selected_kbest
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    clf_report_text = classification_report(y_test, y_pred, target_names=["non egg", "egg"])
    conf_matrix = confusion_matrix(y_test, y_pred)
    pd.DataFrame({
        "true_label": y_test.values,
        "predicted_label": y_pred,
        "predicted_prob": y_proba
    }, index=y_test.index).to_csv("outputs/test_predictions.csv")
    with open("outputs/feature_selection_log.txt", "w") as f:
        f.write("=== Feature Selection Log ===\n\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Total input features: {len(all_features)}\n")
        f.write(f"\nPearson Correlation Threshold: {threshold}\n")
        f.write(f"\nDropped due to correlation ({len(dropped_corr)} features):\n")
        for feat in dropped_corr:
            f.write(f"  - {feat}\n")
        f.write(f"\nRemaining after correlation: {len(features_after_corr)} features\n\n")
        f.write(f"KBest selected ({len(selected_kbest)} features):\n")
        for i, feat in enumerate(selected_kbest):
            score = f"{kbest_scores[i]:.4f}" if kbest_scores is not None else "NA"
            pval = f"{kbest_pvalues[i]:.4e}" if kbest_pvalues is not None else "NA"
            f.write(f"  - {feat} (F-score: {score}, p-value: {pval})\n")
        f.write(f"\nFinal feature count: {len(final_features_used)}\n")
    with open("outputs/performance_summary.txt", "w") as f:
        f.write("=== Model Performance Summary ===\n\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"\nMCC Score: {mcc:.4f}\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"AUC: {auc:.4f}\n")
        f.write("\n--- Classification Report ---\n")
        f.write(clf_report_text)
        f.write("\n--- Confusion Matrix ---\n")
        f.write(np.array2string(conf_matrix, separator=', '))
        f.write("\n\nPipeline Steps:\n")
        for step, obj in model.named_steps.items():
            f.write(f"{step}: {obj.__class__.__name__}\n")
        f.write("\nFinal Model Parameters:\n")
        try:
            for param, val in model.get_params().items():
                f.write(f"{param}: {val}\n")
        except:
            f.write("Model parameters unavailable.\n")
    print("All outputs saved in `outputs/`.")

if __name__ == "__main__":
    main()
