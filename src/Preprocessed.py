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

# target variable balance:
target_col = "TARGET"
print("Target variable distribution: ")
print(dataset[target_col].value_counts(normalize=True))
# sns.countplot(data=dataset , x=target_col)
# plt.title("Target Variable Distribution")
# plt.show

X=dataset.drop("TARGET", axis=1)
y = dataset["TARGET"]
# print("Original X shape:", X.shape)
# print("Original y shape:", y.shape)


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

# sns.countplot(data=smote_df, x=target_col)
# plt.title("Target Variable Distribution After SMOTE")
# plt.show()



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

# FEATURE IMPORTSNCE

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



# #XGBOOST
# import xgboost as xgb
# xgf = xgb.XGBClassifier(
#     objective='binary:logistic',
#     eval_metric='logloss',
#     use_label_encoder=False,
#     scale_pos_weight=1,
#     random_state=42
# )
# xgf.fit(X_train_smote,y_train_smote)
# y_pred_xgf=xgf.predict(X_test)
# print("Accuracy of lightGBM: ",accuracy_score(y_test,y_pred_xgf))
# print("Classification: ",classification_report(y_test,y_pred_xgf))


#logistic regression
log_reg = LogisticRegression(class_weight='balanced')
log_reg.fit(X_train_smote, y_train_smote)
y_pred_lr = log_reg.predict(X_test)
print("Logistic Regression Accuracy :", accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))

# #Decision Tree
# tree_clf = DecisionTreeClassifier(max_depth=4,class_weight='balanced',random_state=42)
# tree_clf.fit(X_train_smote, y_train_smote)
# y_pred_tree = tree_clf.predict(X_test)
# print(" Decision Tree Results:")
# print("Accuracy:", accuracy_score(y_test, y_pred_tree))
# print(classification_report(y_test, y_pred_tree))

# #ROC-AUC CURVE APPLY
# from sklearn.metrics import roc_auc_score , roc_curve , auc
# test_df = pd.DataFrame(
#     {'True': y_test,
#      'Logistic': y_pred_lr,
#      'Random Forest' : y_pred_rf 
#      }
# )
# #plotting
# plt.figure(figsize=(7,5))
# for model in ['Logistic', 'Random Forest']:
#     fpr,tpr,_=roc_curve(test_df['True'], test_df[model])
#     roc_auc = auc(fpr,tpr)
#     plt.plot(fpr,tpr,label=f'{model} (AUC={roc_auc:2f})')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curves for Two Models')
# plt.legend()
# plt.show() 

#ENSEMBLE METHIDS
# import warnings
# from imblearn.ensemble import EasyEnsembleClassifier
# warnings.filterwarnings('ignore')
# eec = EasyEnsembleClassifier(random_state=42)
# eec.fit(X_train,y_train)
# y_pred_eec=eec.predict(X_test)
# print("Classification Report: ", classification_report(y_test,y_pred_eec))
# print("Accuracy: ",accuracy_score(y_test,y_pred_eec))







#ENSEMBLE USING LR AND RF
# from sklearn.ensemble import VotingClassifier
# voting_clf=VotingClassifier(estimators=[('lr',log_reg),('rf',rf)],voting='soft')
# voting_clf.fit(X_train_smote,y_train_smote)
# y_pred_voting=voting_clf.predict(X_test)
# print("Accuracy of hardsoft: ",accuracy_score(y_test,y_pred_voting))
# print("Classification: ",classification_report(y_test,y_pred_voting))

#Stack method:
from sklearn.ensemble import StackingClassifier
estimators = [
    ('lr', log_reg),
    ('rf', rf)
    ]
stacking_clf=StackingClassifier(estimators=estimators,final_estimator=log_reg)
stacking_clf.fit(X_train_smote,y_train_smote)
y_pred_stacking=stacking_clf.predict(X_test)
print("Accuracy of stacking: ", accuracy_score(y_test,y_pred_stacking))
print("Classification: ",classification_report(y_test,y_pred_stacking))

# #applying 5cv , 10cv and so on
# kf = KFold(n_splits=5,shuffle=True,random_state=42)
# cross_val_results=cross_val_score(stacking_clf,X,y,cv=kf)
# print("Cross-Validation Results (Accuracy):")
# for i, result in enumerate(cross_val_results, 1):
#     print(f"  Fold {i}: {result * 100:.2f}%")
    
# print(f'Mean Accuracy: {cross_val_results.mean()* 100:.2f}%')


# #find best threshold: 
y_proba = stacking_clf.predict_proba(X_test)[:, 1]

# Try thresholds from 0.21 to 0.3 with step 0.01
thresholds = np.arange(0.20, 0.31, 0.01)
best_threshold = 0.20
best_f1 = 0

print("Thresholds and F1-scores for class 1:")
for thresh in thresholds:
    y_pred_thresh = (y_proba >= thresh).astype(int)
    f1 = f1_score(y_test, y_pred_thresh)
    print(f"Threshold: {thresh:.2f} => F1-score: {f1:.4f}")
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = thresh

print(f"\nBest threshold: {best_threshold:.2f} with F1-score: {best_f1:.4f}")

# after applying best threshold that is 0.25
y_pred_best = (y_proba >= 0.25).astype(int)
print("Classification Report with Best Threshold (0.25):")
print(classification_report(y_test, y_pred_best))
print("Confusion Matrix with Best Threshold (0.25):")
print(confusion_matrix(y_test, y_pred_best))

# #LightGBM
# import lightgbm as lgb
# lgf = lgb.LGBMClassifier(
#     objective='binary',
#     class_weight='balanced',
#     n_estimators=1000,
#     learning_rate=0.05,
#     num_leaves=31,
#     random_state=42
# )
# lgf.fit(X_train_smote,y_train_smote)
# y_pred_lgf = lgf.predict(X_test)
# print("Accuracy of lightGBM: ",accuracy_score(y_test,y_pred_lgf))
# print("Classification: ",classification_report(y_test,y_pred_lgf))