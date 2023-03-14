
import numpy as np
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score


def LassoModel(X_train, y_train, X_test, y_test):
    # fit the Lasso regression model
    lasso = Lasso()
    params = {'alpha': [0.001, 0.01, 0.1, 1, 10]}
    lasso_cv = GridSearchCV(lasso, params, cv=5)
    lasso_cv.fit(X_train, y_train)

    # find the optimal value of alpha
    lasso_alpha = lasso_cv.best_params_['alpha']

    # evaluate the model on the test data
    # make predictions using Lasso regression model
    lasso_pred = lasso_cv.predict(X_test)

    # calculate the root mean squared error (RMSE) and R-squared value for Lasso regression model
    lasso_rmse = np.sqrt(mean_squared_error(y_test, lasso_pred))
    lasso_r2 = r2_score(y_test, lasso_pred)

    
    print("")
    print("Lasso Regression Model:")
    print("RMSE: ", lasso_rmse)
    print("R-squared value: ", lasso_r2)
    print("")


def RidgeModel(X_train, y_train, X_test, y_test):
    # fit the Ridge regression model
    ridge = Ridge()
    params = {'alpha': [0.001, 0.01, 0.1, 1, 10]}
    ridge_cv = GridSearchCV(ridge, params, cv=5)
    ridge_cv.fit(X_train, y_train)

    # find the optimal value of alpha
    ridge_alpha = ridge_cv.best_params_['alpha']

    # evaluate the model on the test data
    # make predictions using Ridge regression model
    ridge_pred = ridge_cv.predict(X_test)

    # calculate the root mean squared error (RMSE) and R-squared value for Ridge regression model
    ridge_rmse = np.sqrt(mean_squared_error(y_test, ridge_pred))
    ridge_r2 = r2_score(y_test, ridge_pred)

    print("Ridge Regression Model:")
    print("RMSE: ", ridge_rmse)
    print("R-squared value: ", ridge_r2)
    print("")