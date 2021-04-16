#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt


# In[2]:


cases = pd.read_csv('https://raw.githubusercontent.com/ynshung/covid-19-malaysia/master/covid-19-malaysia.csv')[['date','cases']] #to extract data in .csv files ['date', cases] -> to extract data in 'date' and 'cases' rows n columns

cases['date'] = pd.to_datetime(cases['date'], format='%d/%m/%Y') #change the string "date" in csv files to proper format, example "01/01/2020" string to 2020-01-01 -< proper format.

print(cases) #print


# In[3]:


cases['cases'].plot() #to plot the amount of "cases" in csv files


# In[4]:


from statsmodels.graphics.tsaplots import plot_acf
plot_acf(cases['cases']) #to plot autocorrelation


# In[5]:


#for total cases prediction


# In[8]:


X = cases['cases'].values    # values in "cases" being put into arrays of data
train = X[0:426]             # data training
real_data = X[426:]          # value of real data
predictions = []             


# In[9]:


real_data.size                # value of total cases (real data)


# In[10]:


from statsmodels.tsa.arima_model import ARIMA


# In[11]:


#p,d,q     p = periods taken for autoregressive model
# d = order of integration, difference
# q = moving average
model_arima = ARIMA(train, order=(2,1,0))         #creating prediction using module
model_arima_fit = model_arima.fit()
print(model_arima_fit.aic)


# In[12]:


predictions = model_arima_fit.forecast(steps=20)[0]      # prediction data, steps = amount of days for prediction
predictions


# In[13]:


plt.plot(real_data)
plt.plot(predictions,color='red')     # blue line = real data, red line = prediction

plt.xlabel("Days")
plt.ylabel("Total Confirmed Cases")

plt.title("Total Confirmed Cases Prediction")


# In[14]:


#for daily cases change prediction


# In[15]:


daily_change = cases['cases'].diff(periods=1) #daily cases changes periods = 1 means the changes are calculated daily
#integrated of order 1
daily_change = daily_change[1:] #exclude the first day


# In[16]:


print(daily_change)


# In[17]:


plot_acf(daily_change) 


# In[23]:


Y = daily_change.values    # values in "cases" being put into arrays of data
train_new = Y[0:425]             # data training
real_data_2 = Y[425:]          # value of real data
predictions_2 = []


# In[24]:


real_data_2.size


# In[29]:


model_arima = ARIMA(train_new, order=(10,1,0))         #creating prediction using module
model_arima_fit = model_arima.fit()
print(model_arima_fit.aic)


# In[30]:


predictions_2 = model_arima_fit.forecast(steps=20)[0]      # prediction data, steps = amount of days for prediction
predictions_2


# In[31]:


plt.plot(real_data_2)
plt.plot(predictions_2,color='green')    # blue line = real data, green line = prediction


# In[ ]:





# In[ ]:




