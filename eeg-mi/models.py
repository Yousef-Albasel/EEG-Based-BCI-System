from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def get_model_configs():
    return {
        "SVM": (SVC(probability=True), {
            'C': [1, 10],
            'kernel': ['rbf', 'linear'],
        }),
        "RandomForest": (RandomForestClassifier(random_state=42), {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20]
        }),
        "LogisticRegression": (LogisticRegression(max_iter=1000), {
            'C': [0.01, 0.1, 1, 10]
        }),
        "AdaBoost": (AdaBoostClassifier(random_state=42), {
            'n_estimators': [100],
            'learning_rate': [1.0]
        }),
        "XGBoost": (XGBClassifier(eval_metric='logloss'), {
            'n_estimators': [100, 200],
            'max_depth': [3, 5],
            'learning_rate': [0.05, 0.1]
        }),
        "LDA": (LinearDiscriminantAnalysis(), {
        'shrinkage': [None, 'auto'],
        'solver': ['svd', 'lsqr', 'eigen']
        })
    }
