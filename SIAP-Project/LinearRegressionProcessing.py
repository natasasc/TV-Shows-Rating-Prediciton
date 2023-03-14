from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score

def SplitDataSetTrainingAndTest(x,y,testPercent):
    x_train, x_test,y_train,y_test = train_test_split(x,y, test_size = testPercent, random_state = 42)
    return x_train, x_test,y_train,y_test

def SplitDataSet(dataSet,column):
    y = dataSet[column]
    x = dataSet.drop(column, axis=1)

    return x,y


def TrainAndPredict(x_train,y_train,x_test,y_test):
    model = LinearRegression()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print('Linear Regression Model:')
    print("RMSE:",mse)
    print("R-squared value:",r2)
    print("")

    y_pred = y_pred.transpose()

    return y_pred
