# Import relevant library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns

# Load the dataset from sklearn.datasets imported libraries.
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

# Display basic statistics for the original dataset
print("Basic Statistics of the Original Cancer Dataset:")
print(pd.DataFrame(X, columns=cancer.feature_names).describe())

# Pairwise feature correlation heatmap
plt.figure(figsize=(12, 10))
corr_matrix = pd.DataFrame(X, columns=cancer.feature_names).corr()
sns.heatmap(corr_matrix, cmap='coolwarm', annot=False, cbar=True)
plt.title('Correlation Heatmap Feature')
plt.show()

# Standardize the dataset
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reduce dataset into 2 PCA components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Scatter plot for the first two PCA components
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', edgecolor='k', alpha=0.7)
plt.title('PCA of Breast Cancer Dataset')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.colorbar(label='Malignant/Benign')
plt.legend(['Malignant', 'Benign'], loc='upper left')  # Add this line for legend
plt.show()

# Determine the optimal number of PCA components dynamically
pca_full = PCA()
X_pca_full = pca_full.fit_transform(X_scaled)
explained_variance = np.cumsum(pca_full.explained_variance_ratio_)

plt.figure(figsize=(8, 6))
plt.plot(explained_variance, marker='o', linestyle='--')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% Variance Threshold')
plt.title('Cumulative Explained Variance by PCA Components')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.legend()
plt.grid(True)
plt.show()

# Choose the number of components dynamically based on 95% variance
n_components = np.argmax(explained_variance >= 0.95) + 1
print(f"Optimal number of PCA components for 95% variance: {n_components}")

# Apply PCA to the dataset.
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)

# Display feature contributions to principal components
loading_matrix = pca.components_.T * np.sqrt(pca.explained_variance_)
loading_df = pd.DataFrame(
    loading_matrix,
    columns=[f'PC{i+1}' for i in range(n_components)],
    index=cancer.feature_names
)
print("Feature contributions to PCA components:")
print(loading_df)

# Heatmap for feature contributions
plt.figure(figsize=(10, 8))
sns.heatmap(loading_df, cmap='coolwarm', annot=False)
plt.title('Feature Contributions to PCA Components')
plt.show()

# Dataset split for training and testing
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'C': [0.01, 0.1, 1.0, 10],
    'solver': ['liblinear', 'lbfgs']
}
grid_search = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print(f"Best Hyperparameters: {grid_search.best_params_}")

# Train logistic regression model
best_log_reg = grid_search.best_estimator_
best_log_reg.fit(X_train, y_train)

# PCA Model Evaluation
y_pred = best_log_reg.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy of Logistic Regression on PCA-reduced data: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Cross-validation of scores for precision, recall.
cv_precision = cross_val_score(best_log_reg, X_train, y_train, cv=5, scoring='precision').mean()
cv_recall = cross_val_score(best_log_reg, X_train, y_train, cv=5, scoring='recall').mean()
cv_f1 = cross_val_score(best_log_reg, X_train, y_train, cv=5, scoring='f1').mean()
print(f"Cross-Validation Precision: {cv_precision:.2f}")
print(f"Cross-Validation Recall: {cv_recall:.2f}")
print(f"Cross-Validation F1-Score: {cv_f1:.2f}")

# Confusion Matrix for logistic Regression
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=cancer.target_names, yticklabels=cancer.target_names)
plt.title('Confusion Matrix for Logistic Regression')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ROC Curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, best_log_reg.predict_proba(X_test)[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# Comparison with original features
X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
log_reg_orig = LogisticRegression(max_iter=1000).fit(X_train_orig, y_train_orig)
y_pred_orig = log_reg_orig.predict(X_test_orig)
accuracy_orig = accuracy_score(y_test_orig, y_pred_orig)
print(f"Accuracy with Original Features: {accuracy_orig:.2f}")