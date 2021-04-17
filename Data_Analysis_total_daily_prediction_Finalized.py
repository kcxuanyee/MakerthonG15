#!/usr/bin/env python
# coding: utf-8

# In[88]:


import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt


# In[89]:


from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates


# In[90]:


cases = pd.read_csv('https://raw.githubusercontent.com/ynshung/covid-19-malaysia/master/covid-19-malaysia.csv')[['date','cases']] #to extract data in .csv files ['date', cases] -> to extract data in 'date' and 'cases' rows n columns

cases['date'] = pd.to_datetime(cases['date'], format='%d/%m/%Y') #change the string "date" in csv files to proper format, example "01/01/2020" string to 2020-01-01 -< proper format.

print(cases) #print


# In[91]:


cases['cases'].plot() #to plot the amount of "cases" in csv files

plt.xlabel("Days")
plt.ylabel("Total Confirmed Cases")
plt.title("Total Confirmed Cases Prediction")


# In[92]:


from statsmodels.graphics.tsaplots import plot_acf
plot_acf(cases['cases']) #to plot autocorrelation


# In[93]:


#for total cases prediction


# In[94]:


train = cases[0:426]             # data training, put into arrays of data
real_data = cases[426:]          # value of real data
predictions = []


# In[95]:


real_data['cases'].size                # value of total days (real data)


# In[96]:


from statsmodels.tsa.arima_model import ARIMA          #using ARIMA module


# In[97]:


#p,d,q     p = periods taken for autoregressive model
# d = order of integration, difference
# q = moving average
model_arima = ARIMA(train['cases'], order=(2,1,0))         #creating prediction using module, with lowest mean
model_arima_fit = model_arima.fit()
print(model_arima_fit.aic)                                 


# In[98]:


predictions = model_arima_fit.forecast(steps=20)[0]      # prediction data, steps = amount of days for prediction
predictions


# In[99]:


plt.plot(real_data['date'], real_data['cases'])
plt.plot(real_data['date'], predictions ,color='red')     # blue line = real data, red line = prediction

plt.xticks(rotation=90)                  #flip x-axis by 90 degree
plt.xlabel("Days")
plt.ylabel("Total Confirmed Cases")
plt.title("Total Confirmed Cases Prediction")


# In[100]:


files = [real_data, predictions]                              # Create new array
headers = ['Real Data - 20 Days', 'Predictions - 20 Days']    # headers in .csv file
my_df = pd.DataFrame(files).T                                 # using DataFrame to create new list
my_df.columns = headers                                       # putting headers into the list
my_df.to_csv('Real_vs_Predictions_Total.csv', index=False)    # creating new .csv files with data


# In[101]:


my_array = predictions                                         # array for prediction of cases
df = pd.DataFrame(my_array, columns = ['predictions'])        
df                                                             # result


# In[102]:


test = model_arima_fit.plot_predict(1,550)                 # Forecasting data


# In[103]:


df['date'] = pd.date_range(start='16/4/2021', periods=len(df), freq='D')        # Forecasting data with date
df['date'] = df['date'].astype(str)     
df['predictions'] = df['predictions'].astype(int)                               # Forecasting data with date
df


# In[104]:


ddate = df.set_index('date').T.to_dict('list')
ddate


# In[105]:


#for daily cases change prediction


# In[106]:


daily = cases
daily['cases'] = daily['cases'].diff(periods=1) #daily cases changes periods = 1 means the changes are calculated daily
daily = daily[1:]


# In[107]:


daily.head()


# In[108]:


plot_acf(daily['cases']) 


# In[109]:


train_new = daily[0:425]             # data training
real_data_2 = daily[425:]          # value of real data
predictions_2 = []


# In[110]:


real_data_2['cases'].size


# In[111]:


model_arima = ARIMA(train_new['cases'], order=(10,1,0))         #creating prediction using module
model_arima_fit = model_arima.fit()
print(model_arima_fit.aic)


# In[112]:


predictions_2 = model_arima_fit.forecast(steps=20)[0]      # prediction data, steps = amount of days for prediction
predictions_2


# In[113]:


plt.plot(real_data_2['date'], real_data_2['cases'])
plt.plot(real_data_2['date'], predictions_2,color='green')    # blue line = real data, green line = prediction

plt.xticks(rotation=90)
plt.xlabel("Days")
plt.ylabel("Total Daily Cases")
plt.title("Total Daily Cases Prediction")


# In[114]:


files = [real_data_2, predictions_2]
headers = ['Real Data - 20 Days', 'Predictions - 20 Days']
my_df = pd.DataFrame(files).T
my_df.columns = headers
my_df.to_csv('Real_vs_Predictions_Daily.csv', index=False)


# In[117]:


import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import datetime
from datetime import datetime
import pytz

tz = pytz.timezone('Asia/Singapore')
time = datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S')

cred = credentials.Certificate(r"C:\Users\acer\Downloads\covid-19-malaysia-master\makerthong15-firebase-adminsdk-upqzr-b6fa909d61.json")
firebase_admin.initialize_app(cred)
database = firestore.client()


# In[121]:


def save(collection_id, document_id, data) :
    database.collection(collection_id).document(document_id).set(data)
    
save(
    collection_id = "Prediction_Daily Cases",
    document_id = "Until 10th May 2021",
    data = ddate
)

save(
    collection_id = "Prediction_Total Cases",
    document_id = "Until 10th May 2021",
    data = ddate
)

save(
    collection_id = "Total_Death Cases",
    document_id = "Until 10th May 2021",
    data = ddate
)

save(
    collection_id = "Total_Recovery Cases",
    document_id = "Until 10th May 2021",
    data = ddate
)


# In[ ]:




