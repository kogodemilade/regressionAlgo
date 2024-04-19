import pandas as pd
import math, datetime, pickle
import numpy as np
from sklearn import preprocessing, svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from matplotlib import style
import matplotlib.pyplot as plt

date_format = "%Y-%m-%d"
style.use('ggplot')
df = pd.read_csv('archive/GOOG.csv')
df['Date'] = pd.to_datetime(df['Date'], format=date_format)
df.set_index('Date', inplace=True)
df2 = pd.read_csv('archive/GOOG.csv')
df['HL_pct'] = (df['High'] - df['Low']) / df['Low'] * 100
df['Pct_change'] = (df['Adj Close'] - df['Open']) / df['Open'] * 100
df = df[['HL_pct', 'Pct_change', 'Adj Close', 'Volume']]

forecast_col = 'Adj Close'
forecast_out = int(math.ceil(0.01 * len(df)))
# print(forecast_out)
df['Label'] = df[forecast_col].shift(-forecast_out)
X = np.array(df.drop(labels='Label', axis=1))
X = X[:-forecast_out]
X = preprocessing.scale(X)

X_lately = X[-forecast_out:]

df.dropna(inplace=True)

y = np.array(df['Label'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# clf = LinearRegression(n_jobs=10)
# clf.fit(X_train, y_train)
# with open('linearregression.pickle', 'wb') as f:
#     pickle.dump(clf, f)
pickle_in = open('linearregression.pickle', 'rb')
clf = pickle.load(pickle_in)
confidence = clf.score(X_test, y_test)
forecast_set = clf.predict(X_lately)
print(confidence)
df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]

df['Adj Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()