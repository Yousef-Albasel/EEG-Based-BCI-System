from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

def get_model_configs():
    return {
        # "SVM": (SVC(probability=True), {
        #     'C': [0.1, 1, 10],
        #     'kernel': ['rbf', 'linear'],
        #     'gamma': ['scale', 'auto']
        # }),
        # "RandomForest": (RandomForestClassifier(random_state=42), {
        #     'n_estimators': [100, 200],
        #     'max_depth': [None, 10, 20]
        # }),
        # "LogisticRegression": (LogisticRegression(max_iter=1000), {
        #     'C': [0.01, 0.1, 1, 10]
        # }),
        # "GradientBoosting": (GradientBoostingClassifier(), {
        #     'n_estimators': [100, 200],
        #     'learning_rate': [0.05, 0.1],
        #     'max_depth': [3, 5]
        # }),
        "AdaBoost": (AdaBoostClassifier(random_state=42), {
            'n_estimators': [50, 100],
            'learning_rate': [0.5, 1.0]
        }),
        # "XGBoost": (XGBClassifier(use_label_encoder=False, eval_metric='logloss'), {
        #     'n_estimators': [100, 200],
        #     'max_depth': [3, 5],
        #     'learning_rate': [0.05, 0.1]
        # })
    }
