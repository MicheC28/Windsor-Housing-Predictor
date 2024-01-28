from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import pandas as pd

windsor_data_path = "HousePrices.csv"

windsor_data = pd.read_csv(windsor_data_path)

# print(windsor_data.describe())


windsor_features = ["lotsize", 'bedrooms', 'bathrooms', 'stories', 'garage']

X = windsor_data[windsor_features]
y = windsor_data.price


trainX, valX, trainy, valy = train_test_split(X, y, random_state=1)

windsor_model_ranFor = RandomForestRegressor()
windsor_model_ranFor.fit(trainX, trainy)

prediction = windsor_model_ranFor.predict(valX)

prediction_ddf = pd.DataFrame(prediction)

print("Prediction:" + str( prediction))
print("Actual: " + str(valy.head()))

mae_ranfor = mean_absolute_error(valy, prediction)

print('MAE:', mae_ranfor)


# https://towardsdatascience.com/house-prices-prediction-using-deep-learning-dea265cc3154



windsor_model_mulReg = LinearRegression()
windsor_model_mulReg.fit(trainX, trainy)

print("intercept: ", windsor_model_mulReg.intercept_)
print("slope: ", windsor_model_mulReg.coef_)

prediction_mulReg = windsor_model_mulReg.predict(valX)

mae_linReg = mean_absolute_error(valy, prediction_mulReg)

print("MAE-LINREG: ", mae_linReg)

# linReg_ddf = pd.DataFrame({'Actual': valy, 'Predicted': prediction_mulReg})
# top_linReg = linReg_ddf.head()
# top_linReg


#MAE-ranFor: 15553
#MAR-linReg: 14469.481359732505