import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno #helps visualize missing data in a dataset
from sklearn.model_selection import train_test_split , KFold , cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier , plot_tree
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score


filepath= "Datasets/FinalDataset.csv"
dataset = pd.read_csv(filepath)
#drop less featured attribute
dataset=dataset.drop(columns=['REGION_RATING_CLIENT',
    'NAME_HOUSING_TYPE',
    'REGION_RATING_CLIENT_W_CITY',
    'GENDER',
    'FLAG_WORK_PHONE',
    'REG_CITY_NOT_LIVE_CITY',
    'REG_CITY_NOT_WORK_CITY',
    'FLAG_EMAIL',
    'NAME_CONTRACT_TYPE',
    'REG_REGION_NOT_WORK_REGION',
    'FLAG_EMP_PHONE',
    'LIVE_REGION_NOT_WORK_REGION',
    'REG_REGION_NOT_LIVE_REGION',
    'FLAG_CONT_MOBILE',
    'FLAG_MOBIL','NAME_EDUCATION_TYPE',
    'CHILDREN',
    'NAME_TYPE_SUITE',
    'OWN_REALTY',
    'NAME_FAMILY_STATUS',
    'LIVE_CITY_NOT_WORK_CITY'])
print("columns dropped")

#target variable balance:
target_col = "TARGET"
print("Target variable distribution: ")
print(dataset[target_col].value_counts(normalize=True))
sns.countplot(data=dataset , x=target_col)
plt.title("Target Variable Distribution")
plt.show

X=dataset.drop("TARGET", axis=1)
y = dataset["TARGET"]
print("Original X shape:", X.shape)
print("Original y shape:", y.shape)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

#handling data imbalance : SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("After SMOTE:")
print("X_train_smote shape:", X_train_smote.shape)
print("y_train_smote shape:", y_train_smote.shape)

smote_df = pd.DataFrame(X_train_smote)
smote_df[target_col] = y_train_smote
print("Target variable distribution after SMOTE:")
print(smote_df[target_col].value_counts(normalize=True))

sns.countplot(data=smote_df, x=target_col)
plt.title("Target Variable Distribution After SMOTE")
plt.show()

smote_df.to_csv("Datasets/SMOTE_Ready.csv", index=False)

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_smote, y_train_smote)

# import lime
# import lime.lime_tabular
# import numpy as np
# X_train_np = X_train.values
# X_test_np = X_test.values
# explainer = lime.lime_tabular.LimeTabularExplainer(
#     training_data=X_train_np,
#     feature_names=X_train.columns,
#     class_names=['Not Fraud', 'Fraud'],  # or 0/1
#     mode='classification'
# )
# i = 5  
# exp = explainer.explain_instance(
#     X_test_np[i],
#     rf_model.predict_proba,
#     num_features=10
# )
# print(exp.as_list())  
# fig = exp.as_pyplot_figure()
# plt.tight_layout()
# plt.show()

#FEATURE IMPORTSNCE

# importances = rf_model.feature_importances_
# features = X_train_smote.columns
# feat_df = pd.DataFrame({
#     'Feature': features,
#     'Importance': importances
# }).sort_values(by='Importance', ascending=False)
# print("FEATURE IMPORTANCE USING RANDOM FOREST: ",feat_df)

# plt.figure(figsize=(12, 6))
# sns.barplot(data=feat_df.head(20), x="Importance", y="Feature")
# plt.title("Top 20 Important Features (Random Forest)")
# plt.tight_layout()
# plt.show()

#checkking accuracy using different models

#random forest

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight='balanced',
    random_state=42
)
rf.fit(X_train_smote, y_train_smote)
y_pred_rf = rf.predict(X_test)

print("Final Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# #find best threshold: 
# y_proba = rf.predict_proba(X_test)[:, 1]

# # Try thresholds from 0.21 to 0.3 with step 0.01
# thresholds = np.arange(0.21, 0.31, 0.01)
# best_threshold = 0.21
# best_f1 = 0

# print("Thresholds and F1-scores for class 1:")
# for thresh in thresholds:
#     y_pred_thresh = (y_proba >= thresh).astype(int)
#     f1 = f1_score(y_test, y_pred_thresh)
#     print(f"Threshold: {thresh:.2f} => F1-score: {f1:.4f}")
#     if f1 > best_f1:
#         best_f1 = f1
#         best_threshold = thresh

# print(f"\nBest threshold: {best_threshold:.2f} with F1-score: {best_f1:.4f}")

#after applying best threshold that is 0.25
# y_pred_best = (y_proba >= 0.25).astype(int)
# print("Classification Report with Best Threshold (0.25):")
# print(classification_report(y_test, y_pred_best))
# print("Confusion Matrix with Best Threshold (0.25):")
# print(confusion_matrix(y_test, y_pred_best))

# #logistic regression
# log_reg = LogisticRegression(class_weight='balanced')
# log_reg.fit(X_train, y_train)
# y_pred_lr = log_reg.predict(X_test)
# print("Logistic Regression Accuracy :", accuracy_score(y_test, y_pred_lr))
# print(classification_report(y_test, y_pred_lr))

# #Decision Tree
# tree_clf = DecisionTreeClassifier(max_depth=4,class_weight='balanced',random_state=42)
# tree_clf.fit(X_train, y_train)
# y_pred_tree = tree_clf.predict(X_test)
# print(" Decision Tree Results:")
# print("Accuracy:", accuracy_score(y_test, y_pred_tree))
# print(classification_report(y_test, y_pred_tree))

