import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pickle

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')
# Load Dataset
# -----------------------------
file_path = 'C:/Users/user/Downloads/train_u6lujuX_CVtuZ9i.csv'
if not os.path.exists(file_path):
    print("File not found!")
    exit()

df = pd.read_csv(file_path)
print("First rows of data:")
print(df.head())


# -----------------------------
# Preprocessing Function
# -----------------------------
def preprocess_data(df):
    # Fill missing values
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(df[col].median(), inplace=True)

    # Encode categorical features
    encoder = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = encoder.fit_transform(df[col])

    # Feature Engineering
    df['Total_income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
    df['ApplicantIncome_log'] = np.log(df['ApplicantIncome'] + 1)
    df['LoanAmount_log'] = np.log(df['LoanAmount'] + 1)
    df['Total_Income_log'] = np.log(df['Total_income'] + 1)
    df['LoanAmountTerm_log'] = np.log(df['Loan_Amount_Term'] + 1)

    # Drop irrelevant or original columns
    df.drop(columns=['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
                     'Loan_Amount_Term', 'Total_income', 'Loan_ID'], inplace=True)
    return df


df = preprocess_data(df)

# -----------------------------
# Correlation Heatmap
# -----------------------------
plt.figure(figsize=(15, 15))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# -----------------------------
# Features and Target Split
# -----------------------------
X = df.drop(columns=['Loan_Status'])
y = df['Loan_Status']

# -----------------------------
# Train/Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# -----------------------------
# Model Evaluation Function
# -----------------------------
def generate_model_report(name, y_true, y_pred):
    print(f"\n{name} Evaluation:")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("Classification Report:\n", classification_report(y_true, y_pred))


# -----------------------------
# Model Training and Evaluation
# -----------------------------
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "K-Nearest Neighbors": KNeighborsClassifier()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    generate_model_report(name, y_test, y_pred)
    score = cross_val_score(model, X, y, cv=5).mean()
    print(f"Cross-validation Score: {score:.4f}")

# -----------------------------
# Handle Imbalance using SMOTE
# -----------------------------
print("\nApplying SMOTE to balance dataset...")
smote = SMOTE(sampling_strategy='minority', random_state=42)
X_sm, y_sm = smote.fit_resample(X, y)
print("After SMOTE class distribution:\n", pd.Series(y_sm).value_counts())

X_train_sm, X_test_sm, y_train_sm, y_test_sm = train_test_split(X_sm, y_sm, test_size=0.2, random_state=42)

# -----------------------------
# Re-train on Balanced Data
# -----------------------------
best_model = None
best_score = 0

for name, model in models.items():
    model.fit(X_train_sm, y_train_sm)
    y_pred = model.predict(X_test_sm)
    generate_model_report(f"{name} (SMOTE)", y_test_sm, y_pred)
    score = accuracy_score(y_test_sm, y_pred)
    if score > best_score:
        best_score = score
        best_model = model

# -----------------------------
# Save Best Model
# -----------------------------
with open("model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print(f"\nBest model saved as model.pkl with accuracy: {best_score:.2f}")
