import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE

#Custom Wrapper Class
class FraudDetectionModel:
    def __init__(self, model, dropped_cols, threshold=0.5):
        self.model = model
        self.dropped_cols = dropped_cols
        self.threshold = threshold
        self.smote = SMOTE(random_state=42)
        self.trained = False

    def fit(self, X, y):
        X_cleaned = X.drop(columns=self.dropped_cols)
        X_resampled, y_resampled = self.smote.fit_resample(X_cleaned, y)
        self.model.fit(X_resampled, y_resampled)
        self.trained = True

    def predict(self, X):
        if not self.trained:
            raise Exception("Model not trained yet.")
        X_cleaned = X.drop(columns=self.dropped_cols)
        proba = self.model.predict_proba(X_cleaned)[:, 1]
        return (proba >= self.threshold).astype(int)

    def predict_proba(self, X):
        X_cleaned = X.drop(columns=self.dropped_cols)
        return self.model.predict_proba(X_cleaned)

#Columns to Drop
dropped_cols = [
    'REGION_RATING_CLIENT', 'NAME_HOUSING_TYPE', 'REGION_RATING_CLIENT_W_CITY',
    'GENDER', 'FLAG_WORK_PHONE', 'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY',
    'FLAG_EMAIL', 'NAME_CONTRACT_TYPE', 'REG_REGION_NOT_WORK_REGION', 'FLAG_EMP_PHONE',
    'LIVE_REGION_NOT_WORK_REGION', 'REG_REGION_NOT_LIVE_REGION', 'FLAG_CONT_MOBILE',
    'FLAG_MOBIL', 'NAME_EDUCATION_TYPE', 'CHILDREN', 'NAME_TYPE_SUITE', 'OWN_REALTY',
    'NAME_FAMILY_STATUS', 'LIVE_CITY_NOT_WORK_CITY'
]

#Load Data
filepath = "Datasets/FinalDataset.csv"
df = pd.read_csv(filepath)
target_col = "TARGET"

#Define Features & Target
X = df.drop(columns=[target_col])
y = df[target_col]

#Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

#Model Definition
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight='balanced',
    random_state=42
)

# Train using Custom Class with threshold = 0.25
final_model = FraudDetectionModel(model=rf, dropped_cols=dropped_cols, threshold=0.25)
final_model.fit(X_train, y_train)

# Evaluate on Test Set
y_pred = final_model.predict(X_test)
print("Final Evaluation Report:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save Model
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)
joblib.dump(final_model, os.path.join(model_dir, "final_rf_model.pkl"))

print("Final deployable model saved to: models/final_rf_model.pkl")
