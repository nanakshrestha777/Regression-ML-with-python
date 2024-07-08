import pandas as pd
import quandl, math
import numpy as np
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression


df = quandl.get('WIKI/GOOGL')

df = df [['Adj. Open','Adj. High','Adj. Low','Adj. Close', 'Adj. Volume']]

#highlow percentage 
df['HL-PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0

df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

df = df[['Adj. Close','HL-PCT','PCT_change','Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna(-99999, inplace = True)


# ceil rounds up to the nearest int | math.ceil(2.3)~3
# floor rounds down to the nearest int  | math.floor(2.7)~2
forecast_out = int(math.ceil(0.01*len(df)))

df['label'] = df[forecast_col].shift(-forecast_out)

df.dropna(inplace=True)
print(df.head())


x = np.array(df.drop(['label'],axis = 1))
y = np.array(df['label'])



# Print lengths before adjustment
print("Before adjustment:", len(x), len(y)) 

x = preprocessing.scale(x)
x = x[:-forecast_out+1]

# df.dropna(inplace = True)
y = np.array(df['label'])

# Print lengths after adjustment
print(len(x), len(y))


