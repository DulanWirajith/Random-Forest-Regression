import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

"""
# spliting the dataset into test data and training data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
"""

# feature scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)
"""

# Fitting the Random Forest Regression Model to the dataset
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=10000, random_state=0)
regressor.fit(X, y)

# Predicting a new result
vec = 6.5
vec = np.array(vec).reshape(1, -1)
y_pred = regressor.predict(vec)

# Visualising the Decision Tree Regression Model results
# plt.scatter(X, y, color='red')
# plt.plot(X, regressor.predict(X), color='blue')
# plt.title('Truth or Bluff (Decision Tree Regression Model)')
# plt.xlabel('Position level')
# plt.ylabel('Salary')
# plt.show()

# Visualising the Random Forest Regression Model results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.001)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Truth or Bluff (Random Forest Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
