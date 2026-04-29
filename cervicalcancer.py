
#  CERVICAL CANCER PREDICTION — IMPROVED ML PIPELINE
#  Techniques : SMOTE + class_weight + StandardScaler + 8 ML Models
#  Target     : Biopsy (0 = No Cancer, 1 = Cancer)
#  Evaluation : Accuracy, Precision, Recall, F1 Score, Confusion Matrix, ROC

#  WHY THIS APPROACH IS BETTER THAN THE BASIC VERSION:
#  A basic model trained on imbalanced data learns to predict the majority
#  class almost every time — it can score 70%+ accuracy while completely
#  missing most actual cancer cases. In medical diagnosis, missing a cancer
#  case (False Negative) is far more dangerous than a false alarm.
#
#  This pipeline combats that with two complementary strategies:
#
#  1. SMOTE (data-level fix):
#     Synthetically generates new minority-class samples in the training set
#     by interpolating between existing ones. The model sees a balanced dataset
#     during training and learns both classes equally.
#
#  2. class_weight="balanced" (algorithm-level fix):
#     Tells the model to penalise errors on the minority class more heavily.
#     Works as a second layer of protection even if SMOTE isn't perfect.
#
#  3. F1 Score (evaluation fix):
#     Accuracy is misleading on imbalanced data. A model that predicts
#     "Cancer" for everyone gets 57% accuracy but is clinically useless.
#     F1 Score is the harmonic mean of Precision and Recall — it forces
#     the model to be good at BOTH identifying cancer AND avoiding
#     false alarms. It is the correct primary metric here.

# SECTION 1 : IMPORT LIBRARIES 

import warnings
warnings.filterwarnings("ignore")         

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                      
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing & pipeline tools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from imblearn.over_sampling import SMOTE

# Classification models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Evaluation metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)

#SECTION 2 : CONFIGURATION

FILE_PATH   = "Cervical_Cancer_Sample_Dataset.xlsx"   
TARGET_COL  = "Biopsy"
RANDOM_SEED = 42
TEST_SIZE   = 0.20


LEAKAGE_COLS = ["Hinselmann", "Schiller", "Cytology"]

#SECTION 3 : LOAD DATASET 

print("=" * 68)
print("  CERVICAL CANCER PREDICTION — SMOTE + BALANCED WEIGHTS PIPELINE")
print("=" * 68)

try:
    df = pd.read_excel(FILE_PATH)
    print(f"\n Dataset loaded  →  {df.shape[0]:,} rows × {df.shape[1]} columns")
except FileNotFoundError:
    raise FileNotFoundError(
        f"\n '{FILE_PATH}' not found.\n"
        "   Make sure the Excel file is in the same folder as this script."
    )

print(f"\n── Columns : {df.columns.tolist()}")
print(f"\n── Class distribution (before balancing):")
print(df[TARGET_COL].value_counts().rename({0: "No Cancer (0)", 1: "Cancer (1)"}))

#SECTION 4 : HANDLE MISSING VALUES 
df.replace("?", np.nan, inplace=True)


for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

missing_before = df.isnull().sum().sum()
print(f"\n── Missing values detected : {missing_before}")

imputer = SimpleImputer(strategy="median")
df_values = imputer.fit_transform(df)
df = pd.DataFrame(df_values, columns=df.columns)

print(f"── Missing values after median imputation : {df.isnull().sum().sum()} ✅")

# SECTION 5 : FEATURE / TARGET SPLIT

drop_cols = [TARGET_COL] + [c for c in LEAKAGE_COLS if c in df.columns]
X = df.drop(columns=drop_cols)
y = df[TARGET_COL].astype(int)

print(f"\n── Features (X) : {X.shape[1]} columns")
print(f"── Target   (y) : {y.value_counts().to_dict()}")

#SECTION 6 : TRAIN / TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED,
    stratify=y,
)
print(f"\n── Train set : {X_train.shape[0]:,} samples  "
      f"| Class dist: {dict(y_train.value_counts().sort_index())}")
print(f"── Test set  : {X_test.shape[0]:,} samples  "
      f"| Class dist: {dict(y_test.value_counts().sort_index())}")

#SECTION 7 : FEATURE SCALING 

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

#SECTION 8 : APPLY SMOTE ON TRAINING DATA ONLY
smote = SMOTE(random_state=RANDOM_SEED)
X_train_sm, y_train_sm = smote.fit_resample(X_train_scaled, y_train)

print(f"\n── After SMOTE (training data only):")
print(f"   Before → {dict(y_train.value_counts().sort_index())}")
print(f"   After  → {dict(pd.Series(y_train_sm).value_counts().sort_index())} ✅ (balanced)")

#  SECTION 9 : DEFINE ALL 8 MODELS

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",   
        random_state=RANDOM_SEED,
    ),
    "SVM": SVC(
        kernel="rbf",
        probability=True,          
        class_weight="balanced",
        random_state=RANDOM_SEED,
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=6,               
        class_weight="balanced",
        random_state=RANDOM_SEED,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,                 
    ),
    "KNN": KNeighborsClassifier(
        n_neighbors=7,             
    ),
    "Naive Bayes": GaussianNB(),   
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=4,
        random_state=RANDOM_SEED,
    ),
    "AdaBoost": AdaBoostClassifier(
        n_estimators=150,
        learning_rate=0.5,
        random_state=RANDOM_SEED,
    ),
}

# SECTION 10 : HELPER — CONFUSION MATRIX HEATMAP 

def plot_confusion_matrix(cm, model_name, ax):
    """Draw a labelled confusion matrix heatmap on the given Axes object."""
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Cancer", "Cancer"],
        yticklabels=["No Cancer", "Cancer"],
        ax=ax,
        linewidths=0.6,
        linecolor="gray",
        annot_kws={"size": 11},
    )
    ax.set_title(f"{model_name}\nConfusion Matrix", fontsize=10, fontweight="bold", pad=8)
    ax.set_xlabel("Predicted Label", fontsize=9)
    ax.set_ylabel("True Label", fontsize=9)

# SECTION 11 : TRAIN, EVALUATE & VISUALISE 

# Containers to collect results for final plots
summary = {}     
roc_data = {}    


fig_cm,  axes_cm  = plt.subplots(2, 4, figsize=(24, 10))
fig_roc, axes_roc = plt.subplots(2, 4, figsize=(24, 10))
fig_cm.suptitle(
    "Confusion Matrices — All Models (SMOTE + Balanced Weights)",
    fontsize=14, fontweight="bold",
)
fig_roc.suptitle(
    "ROC Curves — All Models (SMOTE + Balanced Weights)",
    fontsize=14, fontweight="bold",
)
axes_cm_flat  = axes_cm.flatten()
axes_roc_flat = axes_roc.flatten()

print("\n" + "=" * 68)
print("  MODEL TRAINING & EVALUATION")
print("=" * 68)

CLASS_NAMES = ["No Cancer (0)", "Cancer (1)"]

for idx, (name, model) in enumerate(models.items()):

    print(f"\n{'─' * 60}")
    print(f"  MODEL {idx + 1} / {len(models)} : {name}")
    print(f"{'─' * 60}")

    # Train on SMOTE-balanced data
    model.fit(X_train_sm, y_train_sm)

    # Predict on the original (unbalanced, unmodified) test set 
    y_pred  = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    # Compute metrics
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)

    summary[name] = {
        "Accuracy" : round(acc  * 100, 2),
        "Precision": round(prec * 100, 2),
        "Recall"   : round(rec  * 100, 2),
        "F1 Score" : round(f1   * 100, 2),
    }

    #Print metrics 
    print(f"\n  Accuracy  : {acc * 100:.2f}%")
    print(f"  Precision : {prec * 100:.2f}%")
    print(f"  Recall    : {rec  * 100:.2f}%")
    print(f"  F1 Score  : {f1   * 100:.2f}%")

 
    
    print(f"\n  Classification Report:\n")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES,
                                zero_division=0))

    #  Confusion Matrix plot 
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, name, axes_cm_flat[idx])

    # ROC Curve plot 
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_score   = auc(fpr, tpr)
    roc_data[name] = (fpr, tpr, auc_score)

    ax = axes_roc_flat[idx]
    ax.plot(fpr, tpr, color="darkorange", lw=2.2,
            label=f"AUC = {auc_score:.4f}")
    ax.fill_between(fpr, tpr, alpha=0.10, color="darkorange")
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random Classifier")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=8)
    ax.set_ylabel("True Positive Rate", fontsize=8)
    ax.set_title(f"{name}\nROC Curve", fontsize=10, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.35)

# Save figure grids 
fig_cm.tight_layout(rect=[0, 0, 1, 0.96])
fig_cm.savefig("confusion_matrices_smote.png", dpi=150, bbox_inches="tight")
plt.close(fig_cm)
print("\n Confusion matrices saved → confusion_matrices_smote.png")

fig_roc.tight_layout(rect=[0, 0, 1, 0.96])
fig_roc.savefig("roc_curves_smote.png", dpi=150, bbox_inches="tight")
plt.close(fig_roc)
print(" ROC curves saved        → roc_curves_smote.png")

# SECTION 12 : OVERLAY ROC — ALL MODELS ON ONE CHART 

PALETTE = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
    "#ff7f00", "#a65628", "#f781bf", "#666666",
]

fig_ov, ax_ov = plt.subplots(figsize=(11, 8))
for (model_name, (fpr, tpr, auc_sc)), color in zip(roc_data.items(), PALETTE):
    ax_ov.plot(fpr, tpr, color=color, lw=2.0,
               label=f"{model_name:<22} AUC = {auc_sc:.4f}")

ax_ov.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Classifier")
ax_ov.set_xlim([0.0, 1.0])
ax_ov.set_ylim([0.0, 1.05])
ax_ov.set_xlabel("False Positive Rate", fontsize=12)
ax_ov.set_ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=12)
ax_ov.set_title(
    "ROC Curve Comparison — All Models\n(SMOTE + Balanced Weights)",
    fontsize=13, fontweight="bold",
)
ax_ov.legend(loc="lower right", fontsize=9, framealpha=0.92,
             prop={"family": "monospace"})
ax_ov.grid(True, linestyle="--", alpha=0.35)
fig_ov.tight_layout()
fig_ov.savefig("roc_overlay_smote.png", dpi=150, bbox_inches="tight")
plt.close(fig_ov)
print(" Overlay ROC saved       → roc_overlay_smote.png")

# SECTION 13 : F1 SCORE COMPARISON BAR CHART 

sorted_by_f1 = dict(
    sorted(summary.items(), key=lambda x: x[1]["F1 Score"], reverse=True)
)
model_names = list(sorted_by_f1.keys())
f1_scores   = [v["F1 Score"] for v in sorted_by_f1.values()]

# Gradient colours: best model bright green, others in a blue-teal range
bar_colors = ["#27ae60"] + ["#2980b9"] * (len(model_names) - 1)

fig_bar, ax_bar = plt.subplots(figsize=(14, 6))
bars = ax_bar.bar(model_names, f1_scores,
                  color=bar_colors, edgecolor="black", linewidth=0.7, width=0.55)

# Annotate each bar with its F1 value
for bar, score in zip(bars, f1_scores):
    ax_bar.text(
        bar.get_x() + bar.get_width() / 2.0,
        bar.get_height() + 0.4,
        f"{score:.2f}%",
        ha="center", va="bottom",
        fontsize=10, fontweight="bold",
    )

mean_f1 = np.mean(f1_scores)
ax_bar.axhline(y=mean_f1, color="red", linestyle="--", linewidth=1.8,
               label=f"Mean F1 = {mean_f1:.2f}%")
ax_bar.set_xlabel("Model", fontsize=12)
ax_bar.set_ylabel("F1 Score (%)", fontsize=12)
ax_bar.set_title(
    "F1 Score Comparison — All Models\n"
    "(Trained with SMOTE + class_weight='balanced')",
    fontsize=13, fontweight="bold",
)
ax_bar.set_ylim([max(0, min(f1_scores) - 8), 105])
ax_bar.tick_params(axis="x", rotation=18)
ax_bar.legend(fontsize=10)
ax_bar.grid(axis="y", linestyle="--", alpha=0.4)
fig_bar.tight_layout()
fig_bar.savefig("f1_comparison_bar_chart.png", dpi=150, bbox_inches="tight")
plt.close(fig_bar)
print("✅ F1 bar chart saved      → f1_comparison_bar_chart.png")

# SECTION 14 : FULL METRICS TABLE 


metrics_df = pd.DataFrame(sorted_by_f1).T  
metrics_df = metrics_df[["Accuracy", "Precision", "Recall", "F1 Score"]]

fig_heat, ax_heat = plt.subplots(figsize=(10, 6))
sns.heatmap(
    metrics_df.astype(float),
    annot=True, fmt=".2f", cmap="YlOrRd",
    linewidths=0.5, linecolor="gray",
    ax=ax_heat, vmin=40, vmax=100,
    annot_kws={"size": 11, "weight": "bold"},
    cbar_kws={"label": "Score (%)"},
)
ax_heat.set_title(
    "All Metrics Heatmap — Accuracy / Precision / Recall / F1\n"
    "(SMOTE + Balanced Weights)",
    fontsize=12, fontweight="bold", pad=12,
)
ax_heat.set_xlabel("Metric", fontsize=11)
ax_heat.set_ylabel("Model", fontsize=11)
ax_heat.tick_params(axis="x", labelsize=10)
ax_heat.tick_params(axis="y", labelsize=9, rotation=0)
fig_heat.tight_layout()
fig_heat.savefig("metrics_heatmap_smote.png", dpi=150, bbox_inches="tight")
plt.close(fig_heat)
print("Metrics heatmap saved   → metrics_heatmap_smote.png")

#  SECTION 15 : FINAL SUMMARY 

best_name = model_names[0]
best_f1   = f1_scores[0]

print("\n" + "=" * 68)
print("  FINAL RESULTS SUMMARY  (sorted by F1 Score — high to low)")
print("=" * 68)
print(f"\n  {'Rank':<5} {'Model':<22} {'Accuracy':>10} {'Precision':>10} "
      f"{'Recall':>8} {'F1 Score':>10}")
print(f"  {'─'*5} {'─'*22} {'─'*10} {'─'*10} {'─'*8} {'─'*10}")

for rank, (mname, scores) in enumerate(sorted_by_f1.items(), start=1):
    tag = "  ← BEST" if rank == 1 else ""
    print(
        f"  {rank:<5} {mname:<22} "
        f"{scores['Accuracy']:>9.2f}% "
        f"{scores['Precision']:>9.2f}% "
        f"{scores['Recall']:>7.2f}% "
        f"{scores['F1 Score']:>9.2f}%{tag}"
    )

print(f"\n  Best Model (by F1 Score) : {best_name}  →  F1 = {best_f1:.2f}%")
print(f"  Mean F1 Score across all : {mean_f1:.2f}%")

print("""
  KEY TAKEAWAYS
  
  • SMOTE balanced the training set — models now learn both classes
    equally instead of defaulting to the majority class.

  • class_weight="balanced" adds a second layer of protection by
    penalising minority-class errors more heavily during training.

  • Recall (sensitivity) is especially important in cancer detection:
    it measures how many actual cancer cases the model correctly
    identifies. Missing a cancer case (False Negative) can be fatal.

  • F1 Score is the right comparison metric because it rewards
    models that achieve high Recall WITHOUT sacrificing Precision.
    A model that shouts "Cancer!" for every patient gets 100% Recall
    but near-0% Precision — F1 Score punishes this behaviour.
  ─────────────────────────────────────────────────────────────────
  Output files saved in this script's folder:
     confusion_matrices_smote.png
     roc_curves_smote.png
     roc_overlay_smote.png
     f1_comparison_bar_chart.png
     metrics_heatmap_smote.png
  ─────────────────────────────────────────────────────────────────
""")