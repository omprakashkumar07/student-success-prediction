# Project name: student-success-prediction
# student dropout prediction (UCI dataset)

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, log_loss,
                             classification_report, ConfusionMatrixDisplay)

# loading the dataset
folder = r"C:\Users\opk62\OneDrive\Desktop\Collage\5th Semester\INT234 - PREDICTIVE ANALYTICS\Project"
df = pd.read_csv(os.path.join(folder, "Dataset", "data.csv"), sep=";")
df.columns = df.columns.str.strip()   # some names had extra spaces

fig_folder = os.path.join(folder, "figures")
os.makedirs(fig_folder, exist_ok=True)

# checking the data first
print("Shape:", df.shape)
print(df["Target"].value_counts())
print("Missing values:", df.isnull().sum().sum())
print("Duplicate rows:", df.duplicated().sum())
print(df.dtypes.value_counts())

# preprocessing

# target is text so converting to numbers
df["Target"] = df["Target"].map({"Dropout": 0, "Enrolled": 1, "Graduate": 2})

# some columns have extreme values, not deleting rows
# just clipping them using IQR so we dont lose data
cont_cols = ["Previous qualification (grade)", "Admission grade",
             "Age at enrollment", "Unemployment rate", "Inflation rate", "GDP"]

for col in cont_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    print(col, "->", ((df[col] < low) | (df[col] > high)).sum(), "outliers clipped")
    df[col] = df[col].clip(low, high)

# new column: approved subjects / enrolled subjects
# if enrolled is 0 it gives nan, so filling with 0
for sem in ["1st", "2nd"]:
    enrolled = df["Curricular units " + sem + " sem (enrolled)"].replace(0, np.nan)
    approved = df["Curricular units " + sem + " sem (approved)"]
    df["Approval rate " + sem + " sem"] = (approved / enrolled).fillna(0)

# these columns are just codes, so one hot encoding
small_cat = ["Marital status", "Application mode", "Course", "Previous qualification"]
big_cat = ["Nacionality", "Mother's qualification", "Father's qualification",
           "Mother's occupation", "Father's occupation"]

# big_cat have too many categories, keeping top 5 and rest as -1
for col in big_cat:
    top5 = df[col].value_counts().nlargest(5).index
    df[col] = df[col].where(df[col].isin(top5), -1)

df = pd.get_dummies(df, columns=small_cat + big_cat, drop_first=True, dtype=int)

print("Shape after preprocessing:", df.shape)
print("Non numeric columns:", df.select_dtypes(exclude="number").columns.tolist())
df.to_csv(os.path.join(folder, "cleaned_data.csv"), index=False)

# EDA
label_names = {0: "Dropout", 1: "Enrolled", 2: "Graduate"}
order = ["Dropout", "Enrolled", "Graduate"]
labels = df["Target"].map(label_names)

key_cols = ["Admission grade", "Age at enrollment",
            "Curricular units 1st sem (grade)", "Curricular units 2nd sem (grade)",
            "Approval rate 1st sem", "Approval rate 2nd sem"]

print(df[key_cols].describe().round(2))
print(df.groupby("Target")[key_cols].mean().round(2).rename(index=label_names).T)

# target is 0,1,2 so correlation on it makes no sense
# made a separate 1/0 column for dropout and used that
dropout_flag = (df["Target"] == 0).astype(int)
corr = df.drop(columns="Target").corrwith(dropout_flag)
corr = corr.sort_values(key=abs, ascending=False)
print("Top 10 correlations with dropout:")
print(corr.head(10).round(3))

# fig 1 - number of students in each class
plt.figure(figsize=(5, 3.5))
sns.countplot(x=labels, order=order)
plt.xlabel("Student outcome")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(fig_folder, "fig1_class_distribution.png"), dpi=300)
plt.show()

# fig 2 - heatmap of top 12 correlated columns
plt.figure(figsize=(8, 6.5))
sns.heatmap(df[corr.head(12).index].corr(), annot=True, fmt=".2f",
            cmap="coolwarm", annot_kws={"size": 6})
plt.tight_layout()
plt.savefig(os.path.join(fig_folder, "fig2_correlation_heatmap.png"), dpi=300)
plt.show()

# fig 3 - 2nd sem grade for each outcome
plt.figure(figsize=(5, 3.5))
sns.boxplot(x=labels, y=df["Curricular units 2nd sem (grade)"], order=order)
plt.xlabel("Student outcome")
plt.tight_layout()
plt.savefig(os.path.join(fig_folder, "fig3_grade_boxplot.png"), dpi=300)
plt.show()

# fig 4 - fees paid or not vs outcome
fees = pd.crosstab(df["Tuition fees up to date"], labels, normalize="index")[order]
fees.plot(kind="bar", stacked=True, figsize=(5, 3.5))
plt.xlabel("Tuition fees up to date (0 = No, 1 = Yes)")
plt.ylabel("Proportion")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(fig_folder, "fig4_tuition_vs_outcome.png"), dpi=300)
plt.show()

# fig 5 - age for each outcome
plt.figure(figsize=(5, 3.5))
sns.kdeplot(x=df["Age at enrollment"], hue=labels, hue_order=order, common_norm=False)
plt.tight_layout()
plt.savefig(os.path.join(fig_folder, "fig5_age_distribution.png"), dpi=300)
plt.show()

# train test split
X = df.drop(columns="Target")
y = df["Target"]

# 80/20 split, stratify so class ratio stays same
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# scaling after split, otherwise test data leaks into training
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# function to get all scores of a model, so i dont repeat the same code 5 times
results = []
predictions = {}

def save_result(name, model, X_te):
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)
    predictions[name] = y_pred
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="macro"),
        "Recall": recall_score(y_test, y_pred, average="macro"),
        "F1 Score": f1_score(y_test, y_pred, average="macro"),
        "AUC": roc_auc_score(y_test, y_prob, multi_class="ovr"),
        "Log Loss": log_loss(y_test, y_prob),
    })
    print(name, "done")

# model 1 - logistic regression
# needs scaled data
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_sc, y_train)
save_result("Logistic Regression", lr, X_test_sc)

# model 2 - KNN
# distance based so scaled data is must
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train_sc, y_train)
save_result("KNN", knn, X_test_sc)

# model 3 - decision tree
# max_depth 6 so it does not overfit, no scaling needed
dt = DecisionTreeClassifier(max_depth=6, random_state=42)
dt.fit(X_train, y_train)
save_result("Decision Tree", dt, X_test)

# model 4 - SVM
# rbf kernel, scaled data
svm = SVC(kernel="rbf", probability=True, random_state=42)
svm.fit(X_train_sc, y_train)
save_result("SVM", svm, X_test_sc)

# model 5 - random forest
# 200 trees, no scaling needed
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
save_result("Random Forest", rf, X_test)

# comparing all models
results_df = pd.DataFrame(results).set_index("Model").round(3)
results_df = results_df.sort_values("F1 Score", ascending=False)
print(results_df.to_string())
results_df.to_csv(os.path.join(folder, "model_results.csv"))

# best model = highest f1
best_name = results_df.index[0]
print("Best model:", best_name)
print(classification_report(y_test, predictions[best_name], target_names=order))

# fig 6 - confusion matrix of all models
fig, axes = plt.subplots(2, 3, figsize=(10, 6.5))
axes = axes.flatten()
for i, (name, y_pred) in enumerate(predictions.items()):
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=["Drop", "Enr", "Grad"],
        normalize="true", cmap="Blues", ax=axes[i],
        colorbar=False, values_format=".2f")
    axes[i].set_title(name, fontsize=9)
axes[5].axis("off")   # only 5 models so last box is empty
plt.tight_layout()
plt.savefig(os.path.join(fig_folder, "fig6_confusion_matrices.png"), dpi=300)
plt.show()

# fig 7 - accuracy and f1 of all models
results_df[["Accuracy", "F1 Score"]].plot(kind="bar", figsize=(7, 3.8))
plt.ylim(0.4, 1)
plt.ylabel("Score")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(fig_folder, "fig7_model_comparison.png"), dpi=300)
plt.show()

# fig 8 - top 10 features used by random forest
imp = pd.Series(rf.feature_importances_, index=X.columns).nlargest(10)
plt.figure(figsize=(7, 4))
imp.sort_values().plot(kind="barh")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(os.path.join(fig_folder, "fig8_feature_importance.png"), dpi=300)
plt.show()