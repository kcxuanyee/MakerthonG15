#!/usr/bin/env python
# coding: utf-8

# In[266]:


import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt


# In[267]:


from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates


# In[268]:


cases = pd.read_csv('https://raw.githubusercontent.com/ynshung/covid-19-malaysia/master/covid-19-malaysia.csv')[['date','cases']] #to extract data in .csv files ['date', cases] -> to extract data in 'date' and 'cases' rows n columns

cases['date'] = pd.to_datetime(cases['date'], format='%d/%m/%Y') #change the string "date" in csv files to proper format, example "01/01/2020" string to 2020-01-01 -< proper format.

print(cases) #print


# In[269]:


cases['cases'].plot() #to plot the amount of "cases" in csv files


# In[270]:


from statsmodels.graphics.tsaplots import plot_acf
plot_acf(cases['cases']) #to plot autocorrelation


# In[271]:


#for total cases prediction


# In[272]:


X = cases['cases'].values    # values in "cases" being put into arrays of data
train = cases[0:426]             # data training
real_data = cases[426:]          # value of real data
predictions = [] 


# In[273]:


real_data['cases'].size                # value of total cases (real data)


# In[274]:


from statsmodels.tsa.arima_model import ARIMA


# In[275]:


#p,d,q     p = periods taken for autoregressive model
# d = order of integration, difference
# q = moving average
model_arima = ARIMA(train['cases'], order=(2,1,0))         #creating prediction using module
model_arima_fit = model_arima.fit()
print(model_arima_fit.aic)


# In[276]:


predictions = model_arima_fit.forecast(steps=20)[0]      # prediction data, steps = amount of days for prediction
predictions


# In[277]:


plt.plot(real_data['date'], real_data['cases'])
plt.plot(real_data['date'], predictions ,color='red')     # blue line = real data, red line = prediction

plt.xticks(rotation=90)
plt.xlabel("Days")
plt.ylabel("Total Confirmed Cases")
plt.title("Total Confirmed Cases Prediction")


# In[278]:


files = [real_data, predictions]    # Create new array
headers = ['Real Data - 20 Days', 'Predictions - 20 Days']    #headers in .csv file
my_df = pd.DataFrame(files).T     #using DataFrame to create new list
my_df.columns = headers           #putting headers into the list
my_df.to_csv('Real_vs_Predictions_Total.csv', index=False)        #creating new .csv files with data


# In[279]:


#for daily cases change prediction


# In[280]:


daily = cases
daily['cases'] = daily['cases'].diff(periods=1) #daily cases changes periods = 1 means the changes are calculated daily
daily = daily[1:]


# In[281]:


daily.head()


# In[282]:


plot_acf(daily['cases']) 


# In[283]:


train_new = daily[0:425]             # data training
real_data_2 = daily[425:]          # value of real data
predictions_2 = []


# In[284]:


real_data_2['cases'].size


# In[285]:


model_arima = ARIMA(train_new['cases'], order=(10,1,0))         #creating prediction using module
model_arima_fit = model_arima.fit()
print(model_arima_fit.aic)


# In[286]:


predictions_2 = model_arima_fit.forecast(steps=20)[0]      # prediction data, steps = amount of days for prediction
predictions_2


# In[287]:


plt.plot(real_data_2['date'], real_data_2['cases'])
plt.plot(real_data_2['date'], predictions_2,color='green')    # blue line = real data, green line = prediction

plt.xticks(rotation=90)
plt.xlabel("Days")
plt.ylabel("Total Daily Cases")
plt.title("Total Daily Cases Prediction")


# In[22]:


files = [real_data_2, predictions_2]
headers = ['Real Data - 20 Days', 'Predictions - 20 Days']
my_df = pd.DataFrame(files).T
my_df.columns = headers
my_df.to_csv('Real_vs_Predictions_Daily.csv', index=False)


# In[ ]:




