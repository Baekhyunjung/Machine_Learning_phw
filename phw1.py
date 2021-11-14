import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import warnings

warnings.filterwarnings(action='ignore')
pd.set_option('display.max_columns', 10)

# Data load
df = pd.read_csv("C:\\Users\\82109\\OneDrive\\문서\\software\\3-2\\ML\\breast-cancer-wisconsin.csv")
# Data preprocessing
df.replace({'?': np.nan}, inplace=True)
print(df.info())
print()
df.dropna(axis=0, how='any', inplace=True)
df['Bare Nuclei'] = df['Bare Nuclei'].astype(np.int64)
print(df.info())
print()
df.drop(['Sample code number'], axis=1, inplace=True)


# Scaling function
def scaling(s, data):
    if (s == 'Standard'):
        scaler = preprocessing.StandardScaler()
        data = scaler.fit_transform(data)
        return data
    elif (s == 'MinMax'):
        scaler = preprocessing.MinMaxScaler()
        data = scaler.fit_transform(data)
        return data
    elif (s == 'Robust'):
        scaler = preprocessing.RobustScaler()
        data = scaler.fit_transform(data)
        return data
    elif (s == 'MaxAbs'):
        scaler = preprocessing.MaxAbsScaler()
        data = scaler.fit_transform(data)
        return data

# Execute GridSearchCV on each model to obtain the best parameters set
def FindBestModel(m, X, y, n):
    # Parameters set of each models
    entropy_param = {'max_depth': [None, 3, 5, 10], 'random_state': [None, 42, 100, 200],
                     'max_features': ['auto', 'sqrt']}
    gini_param = {'max_depth': [None, 3, 5, 10], 'random_state': [None, 42, 100, 200],
                  'max_features': ['auto', 'sqrt']}
    logistic_param = {'penalty': ['l2'], 'random_state': [None, 42, 100, 200],
                      'C': [0.001, 0.01, 0.1, 1, 10, 100],
                      'solver': ['lbfgs', 'sag'], 'max_iter': [10, 100, 200]}
    svc_param = {'kernel': ['linear', 'rbf', 'sigmoid'], 'random_state': [None, 42, 100, 200],
                 'C': [0.01, 0.1, 1, 10, 100],
                 'gamma': ['scale', 'auto'], 'max_iter': [10, 100, 200]}

    # DecisionTreeClassifier (entropy)
    if (m == 'entropy'):
        model = DecisionTreeClassifier(criterion='entropy')
        model_regressor = GridSearchCV(model, entropy_param, scoring='neg_mean_squared_error', cv=n)
        model_regressor.fit(X, y)
        best_param = model_regressor.best_params_
        best_score = model_regressor.best_score_
        print('* Decision Tree Regressor (Entropy) *')
        print('Best parameters: ', best_param)
        print('Score: ', best_score)
        print()
        return best_param, best_score

    # DecisoinTreeClassifier (gini)
    elif (m == 'gini'):
        model = DecisionTreeClassifier(criterion='gini')
        model_regressor = GridSearchCV(model, gini_param, scoring='neg_mean_squared_error', cv=n)
        model_regressor.fit(X, y)
        best_param = model_regressor.best_params_
        best_score = model_regressor.best_score_
        print('* Decision Tree Regressor (Gini) *')
        print('Best parameters: ', best_param)
        print('Score: ', best_score)
        print()
        return best_param, best_score

    # LogisticRegression
    elif (m == 'logistic'):
        model = LogisticRegression()
        model_regressor = GridSearchCV(model, logistic_param, scoring='neg_mean_squared_error', cv=n)
        model_regressor.fit(X, np.ravel(y))
        best_param = model_regressor.best_params_
        best_score = model_regressor.best_score_
        print('* Logistic Regression *')
        print('Best parameters: ', best_param)
        print('Score: ', best_score)
        print()
        return best_param, best_score

    # Support Vector Machine
    elif (m == 'svc'):
        model = SVC()
        model_regressor = GridSearchCV(model, svc_param, scoring='neg_mean_squared_error', cv=n)
        model_regressor.fit(X, np.ravel(y))
        best_param = model_regressor.best_params_
        best_score = model_regressor.best_score_
        print('* Support Vector Machine *')
        print('Best parameters: ', best_param)
        print('Score: ', best_score)
        print()
        print()
        return best_param, best_score


X = df.iloc[:, 0:9] # Predictor variables
y = df[['Class']]   # Target variable

# Sets of scalers, models, and cv values of KFold
scalers = ['Standard', 'MinMax', 'Robust', 'MaxAbs']
models = ['entropy', 'gini', 'logistic', 'svc']
kfold = [3, 5, 10]

for i in scalers:
    print('*****', i, 'Scaling *****')
    X_scaled = scaling(i, X)
    for k in kfold:
        print('*** cv =', k, '***')
        for j in models:
            best_param, best_score = FindBestModel(j, X_scaled, y, k)
