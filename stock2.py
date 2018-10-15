import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
pd.core.common.is_list_like = pd.api.types.is_list_like 
import pandas_datareader.data as web
import quandl
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing, cross_validation, svm
import bs4 as bs
import pickle
import requests
import sys

quandl.ApiConfig.api_key = "RkvCjGeMDmGg5t7rzbAG"

def save_sp500_tickers():
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        tickers.append(ticker)
        
    with open("sp500tickers.pickle","wb") as f:
        pickle.dump(tickers,f)
        
    return tickers

tickers = save_sp500_tickers()
print(tickers)
def linearRegression(ticker):
    style.use('ggplot')
    start = "2018-02-23"
    end = "2018-05-23"

    df = quandl.get(ticker)

    forecast_out = int(30)
    df['Prediction'] = df[['Adj. Close']].shift(-forecast_out)

    X = np.array(df.drop(['Prediction'],1))
    X = preprocessing.scale(X)


    X_forecast = X[-forecast_out:]
    X = X[:-forecast_out]

    y = np.array(df['Prediction'])
    y = y[:-forecast_out]


    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size = 0.2)

    clf = LinearRegression()
    clf.fit(X_train,y_train)

    confidence = clf.score(X_test, y_test)

    print("confidence: ", confidence, '\n')

    forecast_prediction = clf.predict(X_forecast)
    temp = "Linear Regression (30 Day Prediction): " + ticker[5:]
    plt.figure(299)
    plt.plot(forecast_prediction, label = 'Daily Predicted Value')
    plt.suptitle(temp)
    plt.legend(loc='upper right')
    print(forecast_prediction)

def movingAverage(ticker):
    style.use('ggplot')
    plt.figure(300)
    start = "2018-02-23"
    end = "2018-05-23"
    df = quandl.get(ticker, start_date=start, end_date=end)
    df['100ma'] = df['Adj. Close'].rolling(window=100,min_periods=0).mean()

    ax1 = plt.subplot2grid((6,1), (0,0), rowspan=5, colspan=1)
    ax2 = plt.subplot2grid((6,1), (5,0), rowspan=1, colspan=1, sharex=ax1)

    ax1.plot(df.index, df['Adj. Close'], label = 'Adjusted Closing Price')
    ax1.plot(df.index, df['100ma'], label = '100 Day Moving Average')
    ax2.bar(df.index, df['Volume'])
    ax1.legend(loc='upper right')
    temp = "Moving Average: " + ticker[5:]
    plt.suptitle(temp)
    plt.show()

while True:
    var = input("Enter stock ticker: ")
    if var in tickers:
        temp = "WIKI/" + var
        linearRegression(temp)
        movingAverage(temp)
    else:
        var = input("Not a ticker, try again: ")
        if var in tickers:
            temp = "WIKI/" + var
            linearRegression(temp)
            movingAverage(temp)
    


