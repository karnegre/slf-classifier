import numpy as np
import os, json, joblib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, learning_curve
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFECV, f_classif
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix, f1_score, matthews_corrcoef
import joblib
import matplotlib.lines as mlines
from matplotlib.cm import get_cmap
import matplotlib.cm as cm  
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from pprint import pprint
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.feature_selection import RFECV

import warnings
warnings.filterwarnings("ignore")

def preprocess_data(df, threshold, random_state):
    y = df.iloc[:, 0].map({'non egg': 0, 'egg': 1})
    X = df.iloc[:, 1:]

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=random_state)

    corr_matrix = X_train.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), 1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    return X, X_train.drop(columns=to_drop), X_test.drop(columns=to_drop), y_train, y_test

def train_model(X_train, y_train, random_state):
    n_features = X_train.shape[1]-2

    k_values = sorted(set([
        max(1, int(n_features * frac))
        for frac in np.linspace(0.2, 1.0, 4)
    ]))

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('kbest', SelectKBest(score_func=f_classif)),
        ('svc', SVC(class_weight='balanced', probability=True, random_state=random_state))
    ])

    param_grid = [
        {
            'kbest__k': k_values,
            'svc__kernel': ['linear'],
            'svc__C': np.linspace(0.001, 10, 10),
        },
        {
            'kbest__k': k_values,
            'svc__kernel': ['rbf'],
            'svc__C': np.linspace(0.001, 10, 10),
            'svc__gamma': [0.1, 1, 'scale'],
        },
        {
            'kbest__k': k_values,
            'svc__kernel': ['poly'],
            'svc__C': np.linspace(0.001, 10, 10),
            'svc__gamma': [0.1, 1, 'scale'],
            'svc__degree': [2, 3, 4],
            'svc__coef0': [0],
        }
    ]

    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=StratifiedKFold(10, shuffle=True, random_state=random_state),
        scoring='matthews_corrcoef',
        n_jobs=-1,
        verbose=2
    )
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

def plot_combined(cm_train, cm_test, metrics_train, metrics_test, class_names=None, dpi=300):
    blues = cm.get_cmap('Blues')
    greens = cm.get_cmap('Greens')
    color_train = mcolors.to_hex(blues(0.95))
    color_test = mcolors.to_hex(greens(0.95))

    nrows, ncols = cm_train.shape

    if class_names is None:
        class_names = ['non egg', 'egg']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=dpi)

    xticks = class_names
    yticks = class_names

    if nrows == 2 and ncols == 2:
        cell_labels = np.array([["TN", "FP"], ["FN", "TP"]])
    else:
        cell_labels = np.full((nrows, ncols), '', dtype=object)

    ax = axes[0]
    ax.set_xlim(0, ncols)
    ax.set_ylim(0, nrows)
    ax.invert_yaxis()

    ax.set_xticks(np.arange(ncols) + 0.5)
    ax.set_yticks(np.arange(nrows) + 0.5)
    ax.set_xticklabels(xticks, fontsize=16, family='Arial', rotation=45, ha='right')
    ax.set_yticklabels(yticks, fontsize=16, family='Arial')
    ax.tick_params(axis='both', length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for x in range(ncols + 1):
        ax.axvline(x, color='black', lw=1.5)
    for y in range(nrows + 1):
        ax.axhline(y, color='black', lw=1.5)

    for i in range(nrows):
        for j in range(ncols):
            x, y = j, i
            left_half = patches.Rectangle((x, y), 0.5, 1, facecolor=color_train, edgecolor=None)
            right_half = patches.Rectangle((x + 0.5, y), 0.5, 1, facecolor=color_test, edgecolor=None)
            ax.add_patch(left_half)
            ax.add_patch(right_half)

            ax.text(x + 0.5, y + 0.5,
                    f"{cm_train[i, j]} : {cm_test[i, j]}",
                    ha='center', va='center',
                    fontsize=14, fontweight='bold', family='Arial',
                    color='white')

            if cell_labels[i, j]:
                ax.text(x + 0.1, y + 0.1,
                        cell_labels[i, j],
                        ha='left', va='top',
                        fontsize=12, fontweight='bold', family='Arial',
                        color='white')

    ax.set_xlabel("Predicted", fontsize=18, fontweight='bold', family='Arial')
    ax.set_ylabel("True", fontsize=18, fontweight='bold', family='Arial')


    ax = axes[1]
    metrics_df = pd.DataFrame({
        "Metric": list(metrics_train.keys()),
        "Train": list(metrics_train.values()),
        "Validation": list(metrics_test.values())
    }).set_index('Metric')

    bars = metrics_df.plot(kind='barh',
                           width=0.75,
                           color=[color_train, color_test],
                           edgecolor='black',
                           linewidth=0.8,
                           ax=ax,
                           legend=False)

    ax.set_ylabel("", fontsize=18, fontweight='bold', family='Arial')
    ax.set_xlabel("Score (%)", fontsize=18, fontweight='bold', family='Arial')
    ax.tick_params(axis='y', labelsize=14)
    ax.tick_params(axis='x', labelsize=14)
    ax.grid(False)

    for container in bars.containers:
        for bar in container:
            width = bar.get_width()
            ax.text(width - 3,
                    bar.get_y() + bar.get_height() / 2,
                    f'{width:.1f}',
                    ha='right', va='center',
                    fontsize=12,
                    fontweight='bold',
                    color='white' if width > 10 else 'black')

    train_patch = patches.Patch(color=color_train, label='Train')
    test_patch = patches.Patch(color=color_test, label='Validation')
    fig.legend(handles=[train_patch, test_patch],
               title="Data Split",
               fontsize=14,
               title_fontsize=16,
               loc='center right',
               bbox_to_anchor=(1, 0.5),
               borderaxespad=0.)

    plt.tight_layout(rect=[0, 0, 0.85, 1])

def plot_fig1(X_full, X_reduced, output_path="outputs/fig1.png"):

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    corr_full = X_full.corr().abs()
    corr_reduced = X_reduced.corr().abs()

    mask = np.tril(np.ones_like(corr_reduced, dtype=bool))
    step = 10

    plt.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial"],
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    sns.set_style("white")

    fig, axes = plt.subplots(1, 2, figsize=(18, 8), dpi=300)

    sns.heatmap(
        corr_full,
        cmap="GnBu",
        linewidths=0.2,
        vmin=0,
        vmax=1,
        square=True,
        cbar=False,
        ax=axes[0],
        xticklabels=False,
        yticklabels=False,
    )
    axes[0].set_title(f"Original Features: {corr_full.shape[0]}", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Feature Index")
    axes[0].set_ylabel("Feature Index")
    num_full = corr_full.shape[0]
    axes[0].set_xticks(np.arange(0, num_full, step))
    axes[0].set_yticks(np.arange(0, num_full, step))
    axes[0].set_xticklabels(np.arange(0, num_full, step), rotation=45)
    axes[0].set_yticklabels(np.arange(0, num_full, step))

    sns.heatmap(
        corr_reduced,
        mask=mask,
        cmap="GnBu",
        linewidths=0.2,
        vmin=0,
        vmax=1,
        square=True,
        cbar=True,
        cbar_kws={"shrink": 0.7, "label": "|ρ|"},
        ax=axes[1],
        xticklabels=False,
        yticklabels=False,
    )
    axes[1].set_title(f"Reduced Features: {corr_reduced.shape[0]}", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Feature Index")
    axes[1].set_ylabel("")
    num_reduced = corr_reduced.shape[0]
    axes[1].set_xticks(np.arange(0, num_reduced, step))
    axes[1].set_yticks(np.arange(0, num_reduced, step))
    axes[1].set_xticklabels(np.arange(0, num_reduced, step), rotation=45)
    axes[1].set_yticklabels(np.arange(0, num_reduced, step))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

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

def plot_fig4a(grid, out_path="outputs/fig4a.png", dpi=300):
    results = pd.DataFrame(grid.cv_results_)
    best_params = grid.best_params_

    fixed = {'param_kbest__k': best_params['kbest__k']}
    # Remove filtering on degree/gamma to keep all kernels visible
    mask = results['param_kbest__k'] == fixed['param_kbest__k']
    filtered = results[mask]

    plt.figure(figsize=(5, 4), dpi=dpi)
    sns.lineplot(data=filtered, x='param_svc__C', y='mean_test_score',
                 hue='param_svc__kernel', marker='o', linewidth=2)
    for line in plt.gca().lines:
        line.set_markersize(8)
    plt.xlabel("C", fontsize=20, fontweight='bold')
    plt.ylabel("Mean MCC Score", fontsize=20, fontweight='bold')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(title='Kernel', fontsize=14, loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close()

def plot_fig4b(grid, out_path="outputs/fig4b.png", dpi=300):
    results = pd.DataFrame(grid.cv_results_)
    best_params = grid.best_params_
    best_kernel = best_params['svc__kernel']
    kbest_k = best_params['kbest__k']
    kernel_results = results[(results['param_svc__kernel'] == best_kernel) &
                             (results['param_kbest__k'] == kbest_k)].copy()

    plt.figure(figsize=(6, 4), dpi=dpi)

    if best_kernel == "poly":
        kernel_results["gamma_str"] = kernel_results["param_svc__gamma"].astype(str)
        degrees = sorted(kernel_results["param_svc__degree"].unique())
        gammas = sorted(kernel_results["gamma_str"].unique())
        colors = sns.color_palette("viridis", n_colors=len(degrees))
        markers = ['o', 's', '^', 'D', 'v', '*', 'P', 'X', 'h', '+']

        for (deg, gamma), grp in kernel_results.groupby(["param_svc__degree", "gamma_str"]):
            plt.plot(grp["param_svc__C"], grp["mean_test_score"],
                     marker=markers[gammas.index(gamma) % len(markers)],
                     markersize=8, linewidth=2,
                     label=f"deg={deg}, γ={gamma}",
                     color=colors[degrees.index(deg)])

    elif best_kernel == "rbf":
        kernel_results["gamma_str"] = kernel_results["param_svc__gamma"].astype(str)
        gammas = sorted(kernel_results["gamma_str"].unique())
        colors = sns.color_palette("plasma", n_colors=len(gammas))

        for gamma, grp in kernel_results.groupby("gamma_str"):
            x = grp["param_svc__C"].astype(float).values.copy()
            if gamma == "scale":
                x += 0.01
            plt.plot(x, grp["mean_test_score"],
                     marker='o', markersize=6 if gamma == "scale" else 8,
                     linewidth=2, label=f"γ={gamma}", color=colors[gammas.index(gamma)])

    else:
        if "param_svc__C" in kernel_results.columns:
            sorted_res = kernel_results.sort_values("param_svc__C")
            plt.plot(sorted_res["param_svc__C"], sorted_res["mean_test_score"], marker='o', linewidth=2)
        else:
            plt.plot(kernel_results["mean_test_score"], marker='o', linewidth=2)

    plt.xlabel("C", fontsize=20, fontweight='bold')
    plt.ylabel("Mean MCC Score", fontsize=20, fontweight='bold')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True)

    plt.legend(fontsize=14, title_fontsize=14,
               bbox_to_anchor=(1.02, 0.5), loc='center left', borderaxespad=0)

    plt.tight_layout(rect=[0, 0, 0.85, 1])  # leave space on right for legend
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_fig5(best_model, X_train, y_train, out_path="outputs/fig5.png", dpi=300, random_seed=42):
    sizes, tr_scores, val_scores = learning_curve(
        best_model, X_train, y_train,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=StratifiedKFold(10, shuffle=True, random_state=random_seed),
        scoring='matthews_corrcoef', n_jobs=-1
    )

    tr_mean, tr_std = tr_scores.mean(axis=1), tr_scores.std(axis=1)
    val_mean, val_std = val_scores.mean(axis=1), val_scores.std(axis=1)

    plt.figure(figsize=(5, 4), dpi=dpi)
    plt.plot(sizes, tr_mean, 'o-', label='Training', color='royalblue')
    plt.fill_between(sizes, tr_mean - tr_std, tr_mean + tr_std, alpha=0.2, color='royalblue')

    plt.plot(sizes, val_mean, 's--', label='Validation', color='darkorange')
    plt.fill_between(sizes, val_mean - val_std, val_mean + val_std, alpha=0.2, color='darkorange')

    plt.xlabel("Training Set Size", fontsize=18, fontweight='bold')
    plt.ylabel("Mean MCC Score", fontsize=18, fontweight='bold')
    plt.legend(fontsize=15)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close()

def plot_fig6(best_model, X_train, X_test, y_train, y_test, out_path="outputs/fig6.png", dpi=300):
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)

    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_test = confusion_matrix(y_test, y_test_pred)

    metrics_train = {
        "Accuracy": accuracy_score(y_train, y_train_pred) * 100,
        "F1": f1_score(y_train, y_train_pred, average='weighted') * 100,
        "MCC": matthews_corrcoef(y_train, y_train_pred) * 100,
    }
    metrics_test = {
        "Accuracy": accuracy_score(y_test, y_test_pred) * 100,
        "F1": f1_score(y_test, y_test_pred, average='weighted') * 100,
        "MCC": matthews_corrcoef(y_test, y_test_pred) * 100,
    }

    plot_combined(cm_train, cm_test, metrics_train, metrics_test, class_names=["non egg", "egg"], dpi=dpi)

    plt.savefig(out_path, bbox_inches='tight', dpi=dpi)
    plt.close()

def plot_fig7(best_model, X_train, X_test, y_train, y_test, out_path="outputs/fig7.png", dpi=300):
    _roc = lambda y, p: (*roc_curve(y, p, pos_label=1)[:2], auc(*roc_curve(y, p, pos_label=1)[:2]))
    _pr  = lambda y, p: (
        lambda prec, rec: (rec, prec, auc(rec, prec))
    )(*precision_recall_curve(y, p, pos_label=1)[:2])

    y_train_prob = best_model.predict_proba(X_train)[:, 1]
    y_test_prob  = best_model.predict_proba(X_test)[:, 1]

    fpr_tr, tpr_tr, auc_tr = _roc(y_train, y_train_prob)
    fpr_te, tpr_te, auc_te = _roc(y_test , y_test_prob)
    rec_tr, pre_tr, pr_tr  = _pr (y_train, y_train_prob)
    rec_te, pre_te, pr_te  = _pr (y_test , y_test_prob)

    fig, ax = plt.subplots(1, 2, figsize=(9, 4), dpi=dpi)

    ax[0].plot(fpr_tr, tpr_tr, lw=2, color='blue', label=f'Train (AUC = {auc_tr:.3f})')
    ax[0].plot(fpr_te, tpr_te, lw=2, color='green', label=f'Test  (AUC = {auc_te:.3f})')
    ax[0].plot([0, 1], [0, 1], '--', color='gray', lw=1)
    ax[0].set_xlabel("False Positive Rate", fontsize=14, fontweight='bold')
    ax[0].set_ylabel("True Positive Rate", fontsize=14, fontweight='bold')
    ax[0].legend(loc='lower right', fontsize=12)
    ax[0].grid(True, linestyle="--", alpha=0.6)
    ax[0].tick_params(axis='both', labelsize=12)
    ax[0].set_title("ROC Curve", fontsize=16, fontweight="bold")

    ax[1].plot(rec_tr, pre_tr, lw=2, color='blue', label=f'Train (AUC = {pr_tr:.3f})')
    ax[1].plot(rec_te, pre_te, lw=2, color='green', label=f'Test  (AUC = {pr_te:.3f})')
    ax[1].set_xlabel("Recall", fontsize=14, fontweight='bold')
    ax[1].set_ylabel("Precision", fontsize=14, fontweight='bold')
    ax[1].legend(loc='lower left', fontsize=12)
    ax[1].grid(True, linestyle="--", alpha=0.6)
    ax[1].tick_params(axis='both', labelsize=12)
    ax[1].set_title("Precision-Recall Curve", fontsize=16, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close()

def plot_fig8(best_model, X_test, y_test, out_path="outputs/fig8.png", dpi=300, random_seed=42):
    from sklearn.decomposition import PCA
    from sklearn.svm import SVC
    from matplotlib.colors import ListedColormap

    X_scaled = best_model.named_steps["scaler"].transform(X_test)
    X_selected = best_model.named_steps["kbest"].transform(X_scaled)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_selected)

    svc_orig = best_model.named_steps['svc']
    params = svc_orig.get_params()
    params.pop("random_state", None)
    params.pop("class_weight", None)

    svc_2d = SVC(**params, class_weight="balanced", random_state=random_seed)
    svc_2d.fit(X_pca, y_test)

    h = 0.02
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = svc_2d.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    cmap_background = ListedColormap(["#f0c2c2", "#c2f0c2"])
    cmap_points = ListedColormap(["#d62728", "#2ca02c"])

    plt.figure(figsize=(5, 4), dpi=dpi)
    plt.contourf(xx, yy, Z, cmap=cmap_background, alpha=0.5)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_test, cmap=cmap_points,
                edgecolors='k', s=70, linewidth=0.8, alpha=0.9)

    label_names = {0: "non egg", 1: "egg"}
    handles = [plt.Line2D([0], [0], marker='o', color='w',
                          label=label_names[cls], markerfacecolor=cmap_points(cls),
                          markeredgecolor='k', markersize=8)
               for cls in np.unique(y_test)]
    plt.legend(handles=handles, title="Class", fontsize=14, title_fontsize=15, loc='best')

    plt.xlabel("PCA Component 1", fontsize=16, fontweight='bold')
    plt.ylabel("PCA Component 2", fontsize=16, fontweight='bold')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def generate_all_figures(grid, X, X_train, X_test, best_model, y_train, y_test):
    os.makedirs("outputs", exist_ok=True)

    plot_fig2(X, X_train, best_model)
    plot_fig1(X, X_train)
    plot_fig3(grid)
    plot_fig4a(grid)
    plot_fig4b(grid)
    plot_fig5(best_model, X_train, y_train)
    plot_fig6(best_model, X_train, X_test, y_train, y_test)    
    plot_fig7(best_model, X_train, X_test, y_train, y_test)    
    plot_fig8(best_model, X_test, y_test) 

def main():
    # config
    random_state, threshold = 42, 0.65

    # load and process
    df = pd.read_excel("data/data-og.xlsx", engine='openpyxl', sheet_name='dtd_sv_xtrafeat')
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