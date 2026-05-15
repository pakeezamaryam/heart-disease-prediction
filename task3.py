import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Evaluation
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, roc_auc_score

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("dataset_heart.csv")
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())

sns.countplot(x='heart disease', data=df)
plt.xticks([0,1], ['No Disease', 'Disease'])
plt.title("Heart Disease Distribution (Binary)")
plt.show()

sns.histplot(df['age'], bins=20, kde=True)
plt.title("Age Distribution")
plt.show()

df.columns = df.columns.str.strip()
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
print(df.columns)

sns.countplot(x='sex', hue='heart_disease', data=df)
plt.title("Sex vs Heart Disease")
plt.show()

sns.countplot(x='chest_pain_type', hue='heart_disease', data=df)
plt.title("Chest Pain Type vs Heart Disease")
plt.show()

plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

X = df.drop('heart_disease', axis=1)
y = df['heart_disease']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

y_prob = model.predict_proba(X_test)[:, 1]

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label="AUC = " + str(roc_auc))
plt.plot([0, 1], [0, 1], '--')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.coef_[0]
})

importance = importance.sort_values(by="Importance", ascending=False)
print(importance)