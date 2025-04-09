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
sns.countplot(data=dataset , x=target_col)
plt.title("Target Variable Distribution")
plt.show

X=dataset.drop("TARGET", axis=1)
y = dataset["TARGET"]
print("Original X shape:", X.shape)
print("Original y shape:", y.shape)

#SPLITING DATA AS 80-20
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

#RANDOM FOREST 
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_smote, y_train_smote)
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

#logistic regression
log_reg = LogisticRegression(class_weight='balanced')
log_reg.fit(X_train_smote, y_train_smote)
y_pred_lr = log_reg.predict(X_test)
print("Logistic Regression Accuracy :", accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))

#ENSEMBLE USING LR AND RF
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

y_proba = stacking_clf.predict_proba(X_test)[:, 1]
threshold = 0.20
y_pred_thresh = (y_proba >= threshold).astype(int)
print("Accuracy with threshold 0.20: ",accuracy_score(y_test,y_pred_thresh))
print("Classification Report with hreshold (0.20):")
print(classification_report(y_test, y_pred_thresh))
print("Confusion Matrix with Best Threshold (0.20):")
print(confusion_matrix(y_test, y_pred_thresh))

# import lime
import lime.lime_tabular
import numpy as np
X_train_np = X_train.values
X_test_np = X_test.values
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train_np,
    feature_names=X_train.columns,
    class_names=['Not Fraud', 'Fraud'], 
    mode='classification'
)
i = 5  
exp = explainer.explain_instance(
    X_test_np[i],
    stacking_clf.predict_proba,
    num_features=10
)
print(exp.as_list())  
fig = exp.as_pyplot_figure()
plt.tight_layout()
plt.show()

# FEATURE IMPORTSNCE

importances = rf.feature_importances_
features = X_train_smote.columns
feat_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)
print("FEATURE IMPORTANCE USING RANDOM FOREST: ",feat_df)

plt.figure(figsize=(12, 6))
sns.barplot(data=feat_df.head(20), x="Importance", y="Feature")
plt.title("Top 20 Important Features (Random Forest)")
plt.tight_layout()
plt.show()


#ROC-AUC CURVE APPLY BETWEEN LR and RF
from sklearn.metrics import roc_auc_score , roc_curve , auc
test_df = pd.DataFrame(
    {'True': y_test,
     'Logistic': y_pred_lr,
     'Random Forest' : y_pred_rf 
     }
)
#plotting
plt.figure(figsize=(7,5))
for model in ['Logistic', 'Random Forest']:
    fpr,tpr,_=roc_curve(test_df['True'], test_df[model])
    roc_auc = auc(fpr,tpr)
    plt.plot(fpr,tpr,label=f'{model} (AUC={roc_auc:2f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Two Models')
plt.legend()
plt.show() 


#Precision recall curve 
from sklearn.metrics import precision_recall_curve

y_scores = stacking_clf.predict_proba(X_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
auc_score = auc(recall, precision)

plt.figure(figsize=(8,6))
plt.plot(recall, precision, label=f'Precision-Recall Curve (AUC = {auc_score:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()
