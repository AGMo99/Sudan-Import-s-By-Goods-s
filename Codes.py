#Import Packages

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima_model import ARMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


#Data

Data = pd.read_csv(r'C:\Users\AG\Documents\Sudanese Imports by Goods.csv')

Data = Data.drop(columns = 'Unnamed: 0')

#Sum Data

SumData = []

SumData = Data.sum(axis=1)

Use = pd.concat((Data, SumData), 1)

Use = Use.rename(columns={0:'Net'})

#Plot the Data

sns.relplot(x='tea', y='coffee', hue = 'wheat',size=('avo') , data=Data)
sns.catplot(x='lent', y='otherfood',data=Data)
sns.relplot(x='dandt', y='med', hue=('otherch'), data=Data)
sns.catplot(x='petrol', y='otherrowm', data=Data)
sns.relplot(x='mg', y='machin', hue=('transeq'), size=('textils'), data=Data)

#Basic Statistics

Use.describe()

np.max(Use)
np.mean(Use)
np.median(Use)
np.min(Use)
np.std(Use)
np.var(Use)

#Correlation Test

CorrMat = Use.corr()


#ACF and PACF test

netacf = acf(Use['Net'])
netpacf = pacf(Use['Net'])
netpacf = netpacf[0:6]

#Plot the ACF and PACF

plot_acf(Use["Net"])

plot_pacf(Use["Net"])

#Chack for Stationary

sUse = adfuller(Use.Net)
print("ADF Statistics %f" % sUse[0])
print("p-value %f" % sUse[1])
print("Critical Value:")
for key, value in sUse[4].items():
    print('\t%s: %.3f' %(key, value))
    
if sUse[0] < sUse[4]["5%"]:
    print("Reject Ho - Time Series is Stationary")
else:
    print("Failed to Reject Ho - Time series is non-stationary")

#ARMA Model

Model = ARMA(Use['Net'], order=(1, 0))
Model_fit = Model.fit()
Model_fit.summary()

#Predicting

Use['Forecast'] = Model_fit.predict(start=1, end=16, dynamic=True)

#Plot The Model

plt.plot(Use['Forecast'], label='Forecast')
plt.plot(Use['Net'], label='Actual')
plt.title('Forecast Net Import Goods')
plt.legend()
plt.show()

