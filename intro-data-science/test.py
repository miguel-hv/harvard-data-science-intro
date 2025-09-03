# %% 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from prettytable import PrettyTable

# %% 
# Load dataset
filename = 'advertising.csv'  # same as your previous example
df = pd.read_csv(filename)

# Set predictors and target
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# %% 
# Define a grid of alpha values (regularization strengths)
alphas = np.logspace(-3, 3, 100)  # from 0.001 to 1000

# Initialize Ridge regression with cross-validation
ridge_cv = RidgeCV(alphas=alphas, cv=5, scoring='neg_mean_squared_error')  # 5-fold CV

# Fit the model on training data
ridge_cv.fit(X_train, y_train)

# Predict on the test set
y_pred = ridge_cv.predict(X_test)

# Compute the test MSE
test_mse = mean_squared_error(y_test, y_pred)

# %% 
# Print results
print(f'Best alpha found by CV: {ridge_cv.alpha_:.4f}')
print(f'Test MSE: {test_mse:.3f}')

# Pretty table with coefficients
t = PrettyTable(['Predictor', 'Coefficient'])
for coef, col in zip(ridge_cv.coef_, X.columns):
    t.add_row([col, round(coef, 3)])
print(t)

# %% 
# Optional: plot alpha vs. coefficients to see shrinkage effect
plt.figure(figsize=(8,5))
coefs = []
for a in alphas:
    ridge = RidgeCV(alphas=[a], cv=None)
    ridge.fit(X_train, y_train)
    coefs.append(ridge.coef_)

coefs = np.array(coefs)
for i, col in enumerate(X.columns):
    plt.plot(alphas, coefs[:, i], label=col)
plt.xscale('log')
plt.xlabel('Alpha')
plt.ylabel('Coefficient Value')
plt.title('Ridge Coefficients vs Alpha')
plt.legend()
plt.show()
