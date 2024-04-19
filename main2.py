import pandas as pd
import sklearn, math, datetime, pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn import preprocessing, svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from matplotlib import style
import matplotlib.pyplot as plt

n = 0
score = 0
forecast_set = []
avg_list = []
avg_list2 = []

style.use('ggplot')
date_format = "%Y-%m-%d"
df = pd.read_csv('archive/BTCUSD.csv')
df2 = pd.read_csv('archive/BTCUSD.csv')
df['Date'] = pd.to_datetime(df.Date, format=date_format)
df.set_index('Date', inplace=True)

forecast_col = 'Adj Close'
forecast_out = int(math.ceil(0.001 * len(df)))
df['Label'] = df[forecast_col].shift(-forecast_out)
X = np.array(df.drop('Label', axis=1))
X = X[:-forecast_out]
# X = preprocessing.scale(X)
X2 = X[-forecast_out:]
df.dropna(inplace=True)
y = np.array(df['Label'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# clf = LinearRegression(n_jobs=10)
clf = RandomForestRegressor(n_estimators=40, random_state=42, n_jobs=10)
clf2 = GradientBoostingRegressor(n_estimators=40, learning_rate=0.1, random_state=42)
while n <= 5: # To find average
    clf.fit(X_train, y_train)
    clf2.fit(X_train, y_train)
    ## Write to new pickle file, so we don't have to train each run
    # with open('linreg2.pickle', 'wb') as f:
    #     pickle.dump(clf, f)
    pickle_in = open('linreg2.pickle', 'rb')
    # clf = pickle.load(pickle_in)
    forecast_set = clf.predict(X2)
    score = clf.score(X_test, y_test)
    score2 = clf2.score(X_test, y_test)
    avg_list.append(score)
    avg_list2.append(score2)
    n += 1
avg = sum(avg_list)/len(avg_list)
avg2 = sum(avg_list2)/len(avg_list2)
print(f"RF regressor: {avg*100}% \nGB_regressor: {avg2*100}%")
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
