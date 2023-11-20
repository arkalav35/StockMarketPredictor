import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import sys
import matplotlib.pyplot as plt
from textblob import TextBlob
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def predictStockMarket(input_date, input_polarity):
    # Update the file path without any invisible characters
    df1 = pd.read_csv(r'C:\Users\USER\Downloads\NFLX_cp.csv')
    df2 = pd.read_csv(r'C:\Users\user\Downloads\Nflx_news.csv')

    # Check the first few rows of the DataFrame
    df1.head()



    df1.head()

    df2.head()

    d1 = df1.merge(df2, on='Date')

    d1.drop(['News_x'], axis=1)


    l=[]
    x = d1['News_y'].values
    for i in x:
      b=TextBlob(i)
      l.append(b.sentiment.polarity)
      print(l)

    d1 = d1.assign(News_s=l)

    missing_values = d1.isnull().sum()
    print(missing_values)

    d1.head()

    d1.drop(['News_x'], axis=1, inplace=True)

    missing_values = d1.isnull().sum()
    print(missing_values)

    # Calculate IQR for 'Close' column
    Q1 = d1['Close'].quantile(0.25)
    Q3 = d1['Close'].quantile(0.75)
    IQR = Q3 - Q1

    # Define lower and upper bounds to identify outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Identify outliers and remove them
    df_no_outliers = d1[(d1['Close'] >= lower_bound) & (d1['Close'] <= upper_bound)]

    # Calculate IQR for 'News_s' column
    Q1 = d1['News_s'].quantile(0.25)
    Q3 = d1['News_s'].quantile(0.75)
    IQR = Q3 - Q1

    # Define lower and upper bounds to identify outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Identify outliers and remove them
    df_no_outliers = d1[(d1['News_s'] >= lower_bound) & (d1['News_s'] <= upper_bound)]

    d1.head()

    d1.describe()

    # d1.drop(['Open'], axis=1, inplace=True)
    d1.drop(['High'], axis=1, inplace=True)
    d1.drop(['Low'], axis=1, inplace=True)
    d1.drop(['Adj Close'], axis=1, inplace=True)
    d1.drop(['Volume'], axis=1, inplace=True)


    # Assuming 'Date' is the name of your date column
    d1['Date'] = pd.to_datetime(d1['Date'], errors='coerce')
    d1.drop(['News_y'], axis=1, inplace=True)
    d1.set_index('Date', inplace=True)
    d1.head()

    # Assuming your dataset is loaded as 'd1'

    # Convert the 'Date' column to datetime and set it as the index
    # d1['Date'] = pd.to_datetime(d1['Date'])
    # d1.set_index('Date', inplace=True)

    # Optional: Create additional features from the date (e.g., day of week, month, etc.)

    # Convert the 'Date' column to numerical representation (e.g., number of days since the first date in the dataset)
    d1['Days'] = (d1.index - d1.index.min()).days

    # Split the data into features (X) and target variable (y)
    X = d1[['Days', 'News_s']]
    y = d1['Close']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a Linear Regression model
    model = LinearRegression()

    # Check for NaN values in X_train and y_train
    print("NaN values in X_train:")
    print(X_train.isnull().sum())
    print("\nNaN values in y_train:")
    print(y_train.isnull().sum())

    # Impute NaN values in 'Days' column with the mean
    X_train['Days'].fillna(X_train['Days'].mean(), inplace=True)

    # Verify that there are no more NaN values
    print("NaN values in X_train:")
    print(X_train.isnull().sum())

    # Now you can train your model
    model.fit(X_train, y_train)

    # Impute NaN values in 'Days' column with the mean in X_test
    X_test['Days'].fillna(X_test['Days'].mean(), inplace=True)

    # Verify that there are no more NaN values
    print("NaN values in X_test:")
    print(X_test.isnull().sum())

    # Now you can make predictions on X_test
    predictions = model.predict(X_test)

    # Calculate Mean Squared Error to evaluate the model
    mse = mean_squared_error(y_test, predictions)
    print('Mean Squared Error:', mse)
    future_date =  pd.Timestamp(input_date)  # Replace this with the desired future date
    future_days = (future_date - d1.index.min()).days
    future_polarity =  input_polarity  # Provide the polarity of the news headline for the future date

    # Create a DataFrame for prediction
    future_data = pd.DataFrame({'Days': [future_days], 'News_s': [future_polarity]})

    # Use the trained model to make predictions for future data
    future_prediction = model.predict(future_data)

    print(future_prediction[0])
    return future_prediction[0]