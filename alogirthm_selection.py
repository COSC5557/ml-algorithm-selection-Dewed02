import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split
from sklearn.svm import LinearSVR
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

# Read in white wine dataset
wine = pd.read_csv('winequality-white.csv', sep=';')

# Check for any missing values in the dataset
print(f'The data set has missing values: {wine.isnull().values.any()}')

# Assign X to the qualtiy column
y = wine.quality
# Assign y to all other features
X = wine.drop(['quality'], axis=1)

# Split into test and train splits
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Leave unchanged for SVM
X_train_SVM = X_train

# Randomized search using Randomized Forest
rf = RandomForestRegressor()

# Set Ranges for Hyperparameters
n_estimators = [int(x) for x in np.linspace(start=10, stop=100, num=10)]
max_depth = [2, 5, None]
max_features = ['sqrt', 'log2']
bootstrap = [True, False]
min_samples_leaf = [1, 2]

param_distri = {"n_estimators": n_estimators, "max_depth": max_depth, "max_features": max_features,
                "bootstrap": bootstrap, "min_samples_leaf": min_samples_leaf}

# Instantiate Random Search Object for Random Forest
clf_rf = RandomizedSearchCV(estimator=rf, param_distributions=param_distri, cv=5, verbose=2)
clf_rf.fit(X_train, y_train)

# Instantiate Grid Search Object for Random forest
grid_rf = GridSearchCV(estimator=rf, param_grid=param_distri, verbose=2, cv=5)
grid_rf.fit(X_train, y_train)

# Unoptomized Random Forest
un_op_rf = RandomForestRegressor()
un_op_rf.fit(X_train, y_train)

## COULD NOT GET LEARNER TO CONVERGE SO COMMENTED OUT ##
# # SVM with randomized search
# svm = LinearSVR()

# # Set Ranges for Hyperparameters
# # loss = ['epsilon_insensitive', 'squared_epsilon_insensitive']
# # fit_intercept = [True, False]
# # dual = [True]
# # max_iter = [int(x) for x in np.linspace(start=100, stop=1000, num=10)]

# # param_distri_svm = {"loss" : loss, "fit_intercept": fit_intercept, "dual" : dual, "max_iter" : max_iter}
# param_distri_svm = {}
# # Instanitate Random Search Object for SVM
# clf_svm = RandomizedSearchCV(estimator=svm, param_distributions=param_distri_svm, cv=5, verbose=2)

# # Normalize Training data so learner will converge
# scaler = StandardScaler()
# scaler.fit(X_train)
# SVM_train = scaler.transform(X_train)
# SVM_test = scaler.transform(X_test)
# # Fit SVM learner
# clf_svm.fit(SVM_train, y_train)


# Linear Regression model
lin_reg = LinearRegression()

# Set ranges for hyperparamters
fit_intercept = [True, False]
param_distri_lin_reg = {"fit_intercept": fit_intercept}

# Instantiate Randomized Search object for linear regression
clf_lin_reg = RandomizedSearchCV(estimator=lin_reg, param_distributions=param_distri_lin_reg, verbose=2, cv=5)
clf_lin_reg.fit(X_train, y_train)

# Instantiate Grid Search object for linear regression
grid_lin_reg = GridSearchCV(estimator=lin_reg, param_grid=param_distri_lin_reg, verbose=2, cv=5)
grid_lin_reg.fit(X_train, y_train)

# Unoptomized Linear Regression
un_op_lin_reg = LinearRegression()
un_op_lin_reg.fit(X_train, y_train)

# Logistic Regression
log_reg = LogisticRegression()

# Set ranges for hyperparameters
solver = ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']
param_distri_log_reg = {"fit_intercept": fit_intercept, "solver": solver}

# Instantiate Randomized Search object for logistic regression
clf_log_reg = RandomizedSearchCV(estimator=log_reg, param_distributions=param_distri_log_reg, verbose=2, cv=5)
clf_log_reg.fit(X_train, y_train)

# Instantiate Grid Search object for logistic regression
grid_log_reg = GridSearchCV(estimator=log_reg, param_grid=param_distri_log_reg, verbose=2, cv=5)
grid_log_reg.fit(X_train, y_train)

# Unoptomized Logtisic Regression
un_op_log_reg = LogisticRegression()
un_op_log_reg.fit(X_train, y_train)

# KNN
knn = KNeighborsRegressor()

# Set ranges for hyperparameters
n_neighbors = [int(x) for x in np.linspace(start=5, stop=30, num=10)]
weights = ['uniform', 'distance']
algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute']
param_distri_knn = {"n_neighbors": n_neighbors, "weights": weights, "algorithm": algorithm}

# Instantiate Randomized Search object for KNN
clf_knn = RandomizedSearchCV(estimator=knn, param_distributions=param_distri_knn, verbose=2, cv=5)
clf_knn.fit(X_train, y_train)

# Instantiate Grid Search object for KNN
grid_knn = GridSearchCV(estimator=knn, param_grid=param_distri_knn, verbose=2, cv=5)
grid_knn.fit(X_train, y_train)

# Unoptimized KNN
un_op_knn = KNeighborsRegressor()
un_op_knn.fit(X_train, y_train)

# Decision Tree
dtree = DecisionTreeRegressor()

# Set ranges for hyperparameters
criterion = ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
splitter = ['best', 'random']
min_samples_split = [2, 5]
param_distri_dtree = {"criterion": criterion, "splitter": splitter, "min_samples_split": min_samples_split}

# Instantiate Randomized Search object for Decision Tree
clf_dtree = RandomizedSearchCV(estimator=dtree, param_distributions=param_distri_dtree, verbose=2, cv=5)
clf_dtree.fit(X_train, y_train)

# Instantiate Grid Search object for Decision Tree
grid_dtree = GridSearchCV(estimator=dtree, param_grid=param_distri_dtree, verbose=2, cv=5)
grid_dtree.fit(X_train, y_train)

# Unoptomized Decision Tree
un_op_dtree = DecisionTreeRegressor()
un_op_dtree.fit(X_train, y_train)

### RESULTS ###
# Random Forest
print(f'\n\nBest hyperparamters for Random Forest: {clf_rf.best_params_}')
print(f"Random Search Train Accuracy: {clf_rf.score(X_train, y_train):.3}")  # Averaging ~0.98
print(f"Random Search Test Accuracy: {clf_rf.score(X_test, y_test):.3}:")  # Averaging ~0.6
# Optimal hyperparamters are very similar between the two optimization algorithms
print(f'Best hyperparamters for Grid Forest: {grid_rf.best_params_}')
print(f"Grid Search Train Accuracy: {grid_rf.score(X_train, y_train):.3}")  # Averaging ~1.0
print(f"Grid Search Test Accuracy: {grid_rf.score(X_test, y_test):.3}")  # Averaging ~0.6
# Random forest typically withinn a few precent between optomized and unoptomized
print(f'Unoptimized Random Forest Train Accuracy: {un_op_rf.score(X_train, y_train):.3}')  # Averaging ~0.93
print(f'Unoptimized Random Forest Test Accuracy: {un_op_rf.score(X_test, y_test):.3}')  # Averaging ~0.55

# print(f'Best hyperparamters for SVM: {clf_svm.best_params_}')
# print(f"Train Accuray: {clf_svm.score(X_train, y_train):.3}") # Averaging
# print(f"Test Accuray: {clf_svm.score(X_test, y_test):.3}") # Averaging

# Linear Regression
print(f'\nBest hyperparamters Random Search for Linear Regression: {clf_lin_reg.best_params_}')
print(f"Random Search Train Accuracy: {clf_lin_reg.score(X_train, y_train):.3}")  # Averaging ~0.3
print(f"Random Search Test Accuracy: {clf_lin_reg.score(X_test, y_test):.3}")  # Averaging ~0.3
# Best hyperparameters are the same between the two
print(f'Best hyperparamters Grid Search for Linear Regression: {grid_lin_reg.best_params_}')
print(f"Grid Search Train Accuracy: {grid_lin_reg.score(X_train, y_train):.3}")  # Averaging ~o.3
print(f"Grid Search Test Accuracy: {grid_lin_reg.score(X_test, y_test):.3}")  # Averaging ~0.3
# Linear regression appears to produce very similar results regardless of optimization
print(f'Unoptimized Linear Regression Train Accuracy: {un_op_lin_reg.score(X_train, y_train):.3}')  # Averaging ~0.28
print(f'Unoptimized Linear Regression Test Accuracy: {un_op_lin_reg.score(X_test, y_test):.3}')  # Averaging ~0.26

# Logistic Regression
print(f'\nBest hyperparamters Random Search for Logistic Regression: {clf_log_reg.best_params_}')
print(f"Random Search Train Accuracy: {clf_log_reg.score(X_train, y_train):.3}")  # Averaging ~0.5
print(f"Random Search Test Accuracy: {clf_log_reg.score(X_test, y_test):.3}")  # Averaging ~0.5
# Best hyperparameters are the same between the two
print(f'Best hyperparamters Grid Search for Logistic Regression: {grid_log_reg.best_params_}')
print(f"Grid Search Train Accuracy: {grid_log_reg.score(X_train, y_train):.3}")  # Averaging ~0.5
print(f"Grid Search Test Accuracy: {grid_log_reg.score(X_test, y_test):.3}")  # Averaging ~0.5
# # Logistic Regression results are very close typcially just a few precent difference between optomized and unoptimized
print(f'Unoptimized Logistic Regression Train Accuracy: {un_op_log_reg.score(X_train, y_train):.3}')  # Averaging ~0.47
print(f'Unoptimized Logisitc Regression Test Accuracy: {un_op_log_reg.score(X_test, y_test):.3}')  # Averaging ~0.46

# # KNN
print(f'\nBest hyperparamters Random Search for KNN: {clf_knn.best_params_}')
print(f"Random Search Train Accuracy: {clf_knn.score(X_train, y_train):.3}")  # Averaging ~0.99
print(f"Random Search Test Accuracy: {clf_knn.score(X_test, y_test):.3}")  # Averaging ~0.4
# Best hyperparameters are the same between the two
print(f'Best hyperparamters Grid Search for KNN: {grid_knn.best_params_}')
print(f"Grid Search Train Accuracy: {grid_knn.score(X_train, y_train):.3}")  # Averaging ~0.99
print(f"Grid Search Test Accuracy: {grid_knn.score(X_test, y_test):.3}")  # Averaging ~0.4
print(f'Unoptimized KNN Train Accuracy: {un_op_knn.score(X_train, y_train):.3}')  # Averaging ~0.45
print(f'Unoptimized KNN Test Accuracy: {un_op_knn.score(X_test, y_test):.3}')  # Averaging ~0.17

# Decision Tree
print(f'\nBest hyperparamters Random Search for Decision Tree: {clf_dtree.best_params_}')
print(f"Random Search Train Accuracy: {clf_dtree.score(X_train, y_train):.3}")  # Averaging ~0.9
print(f"Random Search Test Accuracy: {clf_dtree.score(X_test, y_test):.3}")  # Averaging ~0.15
# May have different optimal paramters for criterion and splitter
print(f'Best hyperparamters Grid Search for Decision Tree: {grid_dtree.best_params_}')
print(f"Grid Search Train Accuracy: {grid_dtree.score(X_train, y_train):.3}")  # Averaging ~0.9
print(f"Grid Search Test Accuracy: {grid_dtree.score(X_test, y_test):.3}")  # Averaging ~0.15
# Very close results like log reg and random forest however results on testing set is consistently poor
print(f'Unoptimized Decision Tree Train Accuracy: {un_op_dtree.score(X_train, y_train):.3}')  # Averaging ~1.0
print(f'Unoptimized Decision Tree Test Accuracy: {un_op_dtree.score(X_test, y_test):.3}')  # Averaging ~0.1

# Grid Search massively slowed down the run time of the program
# makes sense considering the process of grid search compared to random.
