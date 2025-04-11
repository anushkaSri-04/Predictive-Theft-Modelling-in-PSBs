import joblib
import pandas as pd

class PredictiveModel:
    def __init__(self, model, threshold=0.20):
        self.model = model
        self.threshold = threshold

    def predict(self, input_df):
        # Assume input_df is a preprocessed DataFrame
        probabilities = self.model.predict_proba(input_df)[:, 1]
        predictions = (probabilities >= self.threshold).astype(int)
        return predictions

    def predict_proba(self, input_df):
        return self.model.predict_proba(input_df)

# Load your trained model (from previous training steps)
# Assuming stacking_clf is already trained
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Dummy reinitialization (You would load or reuse the trained stacking_clf)
# For example purposes only; replace this with your real trained stacking_clf
# stacking_clf = joblib.load('trained_Predictive_model.pkl')

# Example initialization (replace with the actual trained model)
log_reg = LogisticRegression()
rf = RandomForestClassifier()
estimators = [('lr', log_reg), ('rf', rf)]
stacking_clf = StackingClassifier(estimators=estimators, final_estimator=log_reg)

# Wrap the trained stacking classifier
final_model = PredictiveModel(model=stacking_clf, threshold=0.20)

# # Save the model for Flask deployment
# joblib.dump(final_model, 'models/Predictive_model.pkl')
# print("Model saved as Predictive_model.pkl")


import pickle



# Save the model for Flask deployment
with open('app/models/Predictive_model.pkl', 'wb') as f:
    pickle.dump(final_model, f)

print("Model saved as Predictive_model.pkl")