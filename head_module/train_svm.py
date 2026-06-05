import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# 1️⃣ Load dataset
df = pd.read_csv("head_module/final_gesture_dataset.csv")

# 2️⃣ Separate features and label
X = df.iloc[:, :-1]   # all columns except last
y = df.iloc[:, -1]    # last column (label)

# 3️⃣ Split dataset (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4️⃣ Create SVM model
model = SVC(kernel='rbf')   # you can try 'linear' also

# 5️⃣ Train model
model.fit(X_train, y_train)

# 6️⃣ Test model
y_pred = model.predict(X_test)

# 7️⃣ Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# 8️⃣ Save model
joblib.dump(model, "models/gesture_svm_model.pkl")

print("\nModel saved as gesture_svm_model.pkl")


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. Full 7x7 Confusion Matrix Data
cm_full = np.array([
    [198, 0, 0, 3, 0, 0, 2],
    [0, 224, 0, 0, 0, 0, 0],
    [0, 0, 178, 0, 0, 0, 0],
    [2, 0, 0, 209, 0, 4, 0],
    [0, 2, 0, 10, 163, 7, 1],
    [0, 0, 0, 6, 21, 154, 30],
    [1, 0, 0, 3, 2, 14, 166]
])

classes = [0, 1, 2, 3, 4, 5, 6]

# --- Visualization ---

plt.figure(figsize=(12, 9))
sns.heatmap(cm_full, annot=True, fmt='d', cmap='YlOrRd', 
            xticklabels=classes, yticklabels=classes)
plt.title('Confusion Matrix - Complete Head Movement Module (7 Classes)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('full_head_confusion_matrix.png', dpi=300)
plt.show()

# 2. Classification Metrics Heatmap
# Based on your 0.92 accuracy report
metrics_full = np.array([
    [0.99, 0.98, 0.98],
    [0.99, 1.00, 1.00],
    [1.00, 1.00, 1.00],
    [0.90, 0.97, 0.94],
    [0.88, 0.89, 0.88],
    [0.86, 0.73, 0.79],
    [0.83, 0.89, 0.86]
])

plt.figure(figsize=(10, 7))
sns.heatmap(metrics_full, annot=True, cmap='RdYlGn',
            xticklabels=['Precision', 'Recall', 'F1-Score'],
            yticklabels=classes)
plt.title('Full Classification Report - Head Movement Metrics')
plt.savefig('full_head_class_report.png', dpi=300)
plt.show()
