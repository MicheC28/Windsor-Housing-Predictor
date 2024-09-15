# House Prices Prediction Project
## Project Description
This project focuses on predicting house prices using two different machine learning models: Random Forest Regressor and Linear Regression. The dataset used contains several features, such as lot size, number of bedrooms, bathrooms, stories, and the presence of a garage. These models are trained to predict the house prices based on these features, and their performances are compared using Mean Absolute Error (MAE) and visualized through error plots.

## Model Comparison and Analysis
Below is a visual representation of the error distribution for both models:
![Error plots for Random Forest Regresor and Linear Regression](https://github.com/MicheC28/Windsor-Housing-Predictor/blob/main/Model_Errors.png?raw=true)
![MAE for Random Forest Regresor and Linear Regression](https://github.com/MicheC28/Windsor-Housing-Predictor/blob/main/MAE.png?raw=true)

## Analysis of the Random Forest Model
The error distribution for the Random Forest model shows a tendency for larger prediction errors as the actual house prices increase. While the model performs reasonably well for lower to mid-range prices (approximately 20,000 to 80,000), it begins to over-predict significantly for more expensive properties. This indicates that the model is struggling to generalize well across the full spectrum of house prices. Overall, the Random Forest model appears to perform adequately for moderately priced homes but demonstrates limitations when handling outliers or high-end properties.

## Analysis of the Linear Regression Model
The Linear Regression model presents a different pattern of prediction errors. As the actual prices rise, the error does too. The model also consistently underestimates more expensive houses.
While there is some clustering around zero error for mid-range prices (40,000 to 80,000), the errors become more evident at lower and higher ends of the price spectrum. 
This suggests that the Linear Regression model, being a simpler model, is underfitting the data. It seems to struggle to model the more complex relationships that may exists in the dataset. 
The model's high bias, especially in under-predicting expensive properties, makes it less effective than the Random Forest model for this specific dataset.

## Overall Model Comparison and Conclusion
When comparing both models, it becomes clear that while both have shortcomings, the Random Forest model performs better for mid-range house prices. 
However, its tendency to overestimate the prices of higher-end properties contributes to its higher MAE. 
Despite the counterintuitive result that Linear Regression has a lower MAE, the scatter plots suggest that the Random Forest model generally captures more complexity in the data. 
Linear Regression consistently underestimates higher prices, pointing to its limitations in this scenario.

To improve the effectiveness of the models, strategies such as feature engineering, hyperparameter tuning for the Random Forest model, or exploring ensemble methods could be employed. 
Overall, the Random Forest model is more effective for predicting mid-range prices, but both models require adjustments to better handle the extremes, particularly high-value properties.

## How to Run the Project
1. Clone the repository and install the required libraries (Pandas, Scikit-learn, Matplotlib).
2. Load the dataset (HousePrices.csv). This dataset is from Kaggle. Feel free to try with prices from other places.
3. Run the script to train both models and display the error plots and MAE comparison.
4. Analyze the output and visualizations to compare the model performance.

## Dependencies
 - Python 3.x
 - Pandas
 - Scikit-learn
 - Matplotlib

## Install dependencies:
```
pip install pandas scikit-learn matplotlib
```

Feel free to clone the repo and experiment with the models!
