import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import (roc_curve, precision_recall_curve, auc, 
                             confusion_matrix, matthews_corrcoef, 
                             f1_score, accuracy_score)

def preprocess_data(df, threshold, random_state):
    y = df.iloc[:, 0].map({'non egg': 0, 'egg': 1})
    X = df.iloc[:, 1:]

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=random_state)

    corr_matrix = X_train.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), 1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    return X, X_train.drop(columns=to_drop), X_test.drop(columns=to_drop), y_train, y_test

def train_model(X_train, y_train, random_state):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('kbest', SelectKBest(score_func=f_classif)),
        ('svc', SVC(class_weight='balanced', probability=True, random_state=random_state))
    ])

    param_grid =[
    {
        'kbest__k': [5, 10, 15, 20],
        'svc__kernel': ['linear'],
        'svc__C': np.linspace(0.001, 10, 10),
    },
    {
        'kbest__k': [5, 10, 15, 20],
        'svc__kernel': ['rbf'],
        'svc__C': np.linspace(0.001, 10, 10),
        'svc__gamma': [0.1, 1, 'scale'],
    },
    {
        'kbest__k': [5, 10, 15, 20],
        'svc__kernel': ['poly'],
        'svc__C': np.linspace(0.001, 10, 10),
        'svc__gamma': [0.1, 1, 'scale'],
        'svc__degree': [2, 3, 4],
        'svc__coef0': [0],
    }
    ]

    grid = GridSearchCV(pipeline, param_grid,
                        cv=StratifiedKFold(5, shuffle=True, random_state=random_state),
                        scoring='matthews_corrcoef', n_jobs=-1, verbose=2)
    grid.fit(X_train, y_train)
    return grid, grid.best_estimator_

def evaluate_model(model, X_train, X_test, y_train, y_test, grid):
    y_test_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    mcc = matthews_corrcoef(y_test, y_test_pred)

    print(f"accuracy: {acc:.4f}, f1-score: {f1:.4f}, mcc: {mcc:.4f}")

    results = pd.DataFrame(grid.cv_results_)
    results.to_csv("outputs/gridsearch_results.csv", index=False)

def plot_fig2(X, X_train, best_model):
    counts = [X.shape[1], X_train.shape[1], best_model.named_steps['kbest'].k]
    plt.figure(figsize=(5, 4), dpi=300)
    bars = plt.bar(["IO Features", "Post-Corr", "KBest"], counts,
                   color=["#3988B7", "#56A36C", "#F2C14E"])
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2, f'{int(bar.get_height())}',
                 ha='center', va='center', fontsize=14, fontweight='bold')
    plt.ylabel("Number of Features", fontsize=16, fontweight='bold')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("outputs/fig2.png", dpi=300)
    plt.close()

def plot_fig3(grid):
    results = pd.DataFrame(grid.cv_results_)
    best_params = grid.best_params_
    svc_params = {k: best_params[k] for k in best_params if 'svc' in k}
    mask = np.all([results[f'param_{k}'] == v for k, v in svc_params.items()], axis=0)
    results['param_kbest__k'] = pd.to_numeric(results['param_kbest__k'], errors='coerce')
    k_vs_auc = results[mask].sort_values('param_kbest__k')

    plt.figure(figsize=(5, 4), dpi=300)
    sns.lineplot(x='param_kbest__k', y='mean_test_score', data=k_vs_auc,
                 marker='o', linewidth=2)
    plt.xlabel("k Features", fontsize=16, fontweight='bold')
    plt.ylabel("Mean MCC Score", fontsize=16, fontweight='bold')
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig("outputs/fig3.png", dpi=300)
    plt.close()

def plot_fig4a(grid):
    results = pd.DataFrame(grid.cv_results_)
    best_params = grid.best_params_

    fixed = {'param_kbest__k': best_params['kbest__k']}
    if 'svc__degree' in best_params:
        fixed['param_svc__degree'] = best_params['svc__degree']
    if 'svc__gamma' in best_params:
        fixed['param_svc__gamma'] = best_params['svc__gamma']

    mask = results['param_kbest__k'] == fixed['param_kbest__k']
    if 'param_svc__degree' in fixed:
        mask &= results['param_svc__degree'] == fixed['param_svc__degree']
    if 'param_svc__gamma' in fixed:
        mask &= results['param_svc__gamma'] == fixed['param_svc__gamma']
    filtered = results[mask].copy()

    plt.figure(figsize=(5, 4), dpi=300)
    sns.lineplot(data=filtered, x='param_svc__C', y='mean_test_score',
                 hue='param_svc__kernel', marker='o', linewidth=2)
    for line in plt.gca().lines:
        line.set_markersize(8)
    plt.legend(title='Kernel', fontsize=12, title_fontsize=12, loc='best')
    plt.xlabel("C", fontsize=16, fontweight='bold')
    plt.ylabel("Mean MCC Score", fontsize=16, fontweight='bold')
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig("outputs/fig4a.png", dpi=300)
    plt.close()


def plot_fig4b(grid):
    results = pd.DataFrame(grid.cv_results_)
    best_params = grid.best_params_
    best_kernel = best_params['svc__kernel']
    kbest_k = best_params['kbest__k']
    kernel_results = results[(results['param_svc__kernel'] == best_kernel) &
                             (results['param_kbest__k'] == kbest_k)].copy()

    plt.figure(figsize=(5, 4), dpi=300)

    if best_kernel == 'poly':
        kernel_results['gamma_str'] = kernel_results['param_svc__gamma'].astype(str)
        degrees = sorted(kernel_results['param_svc__degree'].unique())
        gammas = sorted(kernel_results['gamma_str'].unique())
        colors = sns.color_palette("viridis", n_colors=len(degrees))
        markers = ['o', 's', '^', 'D', 'v', '*', 'P', 'X', 'h', '+']

        for (deg, gamma), grp in kernel_results.groupby(['param_svc__degree', 'gamma_str']):
            plt.plot(grp['param_svc__C'], grp['mean_test_score'],
                     marker=markers[gammas.index(gamma) % len(markers)],
                     markersize=8, linewidth=2,
                     label=f"deg={deg}, γ={gamma}",
                     color=colors[degrees.index(deg)])

    elif best_kernel == 'rbf':
        kernel_results['gamma_str'] = kernel_results['param_svc__gamma'].astype(str)
        gammas = sorted(kernel_results['gamma_str'].unique())
        colors = sns.color_palette("plasma", n_colors=len(gammas))

        for gamma, grp in kernel_results.groupby('gamma_str'):
            x = grp['param_svc__C'].values.copy()
            y = grp['mean_test_score'].values
            if gamma == 'scale':
                x += 0.01
            plt.plot(x, y, marker='o', markersize=6 if gamma == 'scale' else 8,
                     linewidth=2, label=f"γ={gamma}", color=colors[gammas.index(gamma)])

    elif best_kernel == 'linear':
        sorted_res = kernel_results.sort_values('param_svc__C')
        plt.plot(sorted_res['param_svc__C'], sorted_res['mean_test_score'],
                 marker='o', linewidth=2)

    else:
        if 'param_svc__C' in kernel_results.columns:
            sorted_res = kernel_results.sort_values('param_svc__C')
            plt.plot(sorted_res['param_svc__C'], sorted_res['mean_test_score'],
                     marker='o', linewidth=2)
        else:
            plt.plot(kernel_results['mean_test_score'], marker='o', linewidth=2)

    plt.xlabel("C", fontsize=16, fontweight='bold')
    plt.ylabel("Mean MCC Score", fontsize=16, fontweight='bold')
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=12, title_fontsize=12)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig("outputs/fig4b.png", dpi=300)
    plt.close()

def plot_fig5(best_model, X_train, y_train):
    train_sizes, train_scores, val_scores = learning_curve(
        best_model, X_train, y_train,
        cv=5, scoring="matthews_corrcoef", n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 5), random_state=42
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    plt.figure(figsize=(5, 4), dpi=300)
    plt.plot(train_sizes, train_mean, 'o-', color='royalblue', label='Training MCC')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2, color='royalblue')

    plt.plot(train_sizes, val_mean, 'o-', color='darkorange', label='Validation MCC')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2, color='darkorange')

    plt.title("Learning Curve", fontsize=16, fontweight="bold")
    plt.xlabel("Training Set Size", fontsize=14, fontweight='bold')
    plt.ylabel("Mean MCC Score", fontsize=14, fontweight='bold')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("outputs/fig5.png", dpi=300)
    plt.close()

def plot_fig6(best_model, X_test, y_test):
    y_pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    labels = ["non egg", "egg"]
    fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=labels, yticklabels=labels,
                annot_kws={"fontsize": 14, "fontweight": "bold"})
    ax.set_xlabel("Predicted", fontsize=16, fontweight="bold")
    ax.set_ylabel("Actual", fontsize=16, fontweight="bold")
    ax.set_title(f"Test Confusion Matrix\nAcc={acc:.3f}, F1={f1:.3f}, MCC={mcc:.3f}",
                 fontsize=14, fontweight="bold")
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig("outputs/fig6.png", dpi=300)
    plt.close()


def plot_fig7(best_model, X_test, y_test):
    decision_scores = best_model.decision_function(X_test)
    fpr, tpr, _ = roc_curve(y_test, decision_scores)
    roc_auc = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(y_test, decision_scores)
    pr_auc = auc(recall, precision)

    fig, axs = plt.subplots(1, 2, figsize=(9, 4), dpi=300)

    axs[0].plot(fpr, tpr, color='blue', lw=2, label=f'AUC = {roc_auc:.3f}')
    axs[0].plot([0, 1], [0, 1], linestyle='--', color='gray')
    axs[0].set_title("ROC Curve", fontsize=16, fontweight="bold")
    axs[0].set_xlabel("False Positive Rate", fontsize=14, fontweight='bold')
    axs[0].set_ylabel("True Positive Rate", fontsize=14, fontweight='bold')
    axs[0].legend(loc="lower right", fontsize=12)
    axs[0].grid(True, linestyle="--", alpha=0.6)

    axs[1].plot(recall, precision, color='darkorange', lw=2, label=f'AUC = {pr_auc:.3f}')
    axs[1].set_title("Precision-Recall Curve", fontsize=16, fontweight="bold")
    axs[1].set_xlabel("Recall", fontsize=14, fontweight='bold')
    axs[1].set_ylabel("Precision", fontsize=14, fontweight='bold')
    axs[1].legend(loc="lower left", fontsize=12)
    axs[1].grid(True, linestyle="--", alpha=0.6)

    for ax in axs:
        ax.tick_params(axis='both', labelsize=12)

    plt.tight_layout()
    plt.savefig("outputs/fig7.png", dpi=300)
    plt.close()

def plot_fig8(best_model, X_test, y_test):
    X_scaled = best_model.named_steps["scaler"].transform(X_test)
    X_selected = best_model.named_steps["kbest"].transform(X_scaled)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_selected)

    label_names = {0: "non egg", 1: "egg"}
    df_plot = pd.DataFrame({
        "PC1": X_pca[:, 0],
        "PC2": X_pca[:, 1],
        "Label": [label_names[y] for y in y_test]
    })

    palette = sns.color_palette(["#d62728", "#2ca02c"])

    plt.figure(figsize=(5, 4), dpi=300)
    sns.scatterplot(
        data=df_plot, x="PC1", y="PC2", hue="Label",
        palette=palette, s=70, edgecolor="black", alpha=0.9
    )
    plt.title("PCA of Selected Features", fontsize=18, fontweight="bold")
    plt.xlabel("Principal Component 1", fontsize=16, fontweight='bold')
    plt.ylabel("Principal Component 2", fontsize=16, fontweight='bold')
    plt.legend(fontsize=14, title="Class", title_fontsize=15, loc='best')
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig("outputs/fig8.png", dpi=300)
    plt.close()


def generate_all_figures(grid, X, X_train, X_test, best_model, y_train, y_test):
    os.makedirs("outputs", exist_ok=True)

    plot_fig2(X, X_train, best_model)
    plot_fig3(grid)
    plot_fig4a(grid)
    plot_fig4b(grid)
    plot_fig5(best_model, X_train, y_train)
    plot_fig6(best_model, X_test, y_test)    
    plot_fig7(best_model, X_test, y_test)    
    plot_fig8(best_model, X_test, y_test) 

def main():
    # config
    random_state, threshold = 42, 0.55

    # load and process
    df = pd.read_excel("data/data.xlsx", engine='openpyxl')
    X, X_train, X_test, y_train, y_test = preprocess_data(df, threshold, random_state)

    # train model
    grid, best_model = train_model(X_train, y_train, random_state)

    # evaluate and save results
    evaluate_model(best_model, X_train, X_test, y_train, y_test, grid)

    # generate plots
    generate_all_figures(grid, X, X_train, X_test, best_model, y_train, y_test)

    os.makedirs("outputs", exist_ok=True)
    joblib.dump(best_model, "outputs/final_model.pkl")

if __name__ == "__main__":
    main()