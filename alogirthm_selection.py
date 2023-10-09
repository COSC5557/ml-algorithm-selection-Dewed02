import pandas as pd
import numpy as np 
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split

# Read in white wine dataset 
wine = pd.read_csv('winequality-white.csv', sep=';')

# Check for any missing values in the dataset 
print(f'The data set has missing values: {wine.isnull().values.any()}')

# Assign X to the qualtiy column 
X = wine.quality
# Assign y to all other features
y = wine.drop(['quality'], axis=1)

# Split into test and train splits
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# Reshape X vectors
X_train = X_train.to_numpy().reshape(-1,1)
X_test = X_test.to_numpy().reshape(-1,1)

# Randomized search using Randomized Forest
rf = RandomForestRegressor()

# Set Ranges for Hyperparameters 
n_estimators = [int(x) for x in np.linspace(start=10, stop=400, num=10)]
max_depth = [2, 5, None]
max_features = ['sqrt', 'log2']
bootstrap = [True, False]
min_samples_leaf = [1, 2]

param_distri = {"n_estimators" : n_estimators,"max_depth" : max_depth, "max_features" : max_features, "bootstrap" : bootstrap,  "min_samples_leaf" : min_samples_leaf }

# Intantiate Random Search Object 
clf = RandomizedSearchCV(estimator=rf,param_distributions=param_distri, cv=5, verbose=2)
clf.fit(X_train, y_train)
print(clf.best_params_)
print(f"Train Accuray: {clf.score(X_train, y_train):.3}") # Averaging ~0.05
print(f"Test Accuray: {clf.score(X_test, y_test):.3}") # Averaging ~0.05

# It states in the project description that results don't matter, but these seem especially bad. 
# Do you have any suggestion on what I could try to improve the results. 