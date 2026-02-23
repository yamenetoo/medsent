from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

def get_ml_models():
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'SVM_linear': SVC(kernel='linear', probability=True, random_state=42),
        'SVM_rbf': SVC(kernel='rbf', probability=True, random_state=42),
        'RandomForest': RandomForestClassifier(random_state=42),
        'XGBoost': xgb.XGBClassifier(eval_metric='mlogloss', random_state=42),
        'NaiveBayes': MultinomialNB(),
        'kNN': KNeighborsClassifier(metric='cosine'),
        'LDA': LinearDiscriminantAnalysis()
    }
    param_grids = {
        'LogisticRegression': {'C': [0.01, 0.1, 1, 10]},
        'SVM_linear': {'C': [0.1, 1, 10]},
        'SVM_rbf': {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']},
        'RandomForest': {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20],
                         'min_samples_split': [2, 5]},
        'XGBoost': {'n_estimators': [50, 100], 'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1]},
        'NaiveBayes': {'alpha': [0.1, 0.5, 1.0]},
        'kNN': {'n_neighbors': [5]},
        'LDA': {'solver': ['svd']}
    }
    return models, param_grids

def train_ml_model(model, param_grid, X_train, y_train, cv=5):
    if param_grid:
        gs = GridSearchCV(model, param_grid, cv=cv, scoring='f1_macro', n_jobs=-1)
        gs.fit(X_train, y_train)
        return gs.best_estimator_
    else:
        model.fit(X_train, y_train)
        return model