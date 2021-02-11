# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score


# In[2]:


data = pd.read_csv('flights.csv', low_memory = False)
data.head()


# In[3]:


len(data)


# In[4]:


data = data[:100000]
len(data)


# In[5]:


data.head()


# In[6]:


data.info()


# In[7]:


data["DIVERTED"].value_counts()


# In[8]:


data.columns


# In[9]:


data.head()


# In[10]:


# sns.pairplot(data)


# In[11]:


data.columns


# In[12]:


data = data.drop([
    'YEAR',
    'FLIGHT_NUMBER',
    'AIRLINE',
    'DISTANCE',
    'TAIL_NUMBER',
    'TAXI_OUT',
    'SCHEDULED_TIME',
    'DEPARTURE_TIME',
    'WHEELS_OFF',
    'ELAPSED_TIME',
    'AIR_TIME',
    'WHEELS_ON',
    'DAY_OF_WEEK',
    'TAXI_IN',
    'ARRIVAL_TIME',
    'CANCELLATION_REASON'
], axis = 1)
data.head()


# In[13]:


data.isnull().sum()


# In[14]:


data['DEPARTURE_DELAY'] = data['DEPARTURE_DELAY'].fillna(data['DEPARTURE_DELAY'].mean())
data['ARRIVAL_DELAY'] = data['ARRIVAL_DELAY'].fillna(data['ARRIVAL_DELAY'].mean())


# In[15]:


data.head()


# In[16]:


data.isnull().sum()


# In[17]:


data['ARRIVAL_DELAY']


# In[18]:


def func(i):
    if(i>15):
        return 1
    return 0
data['result'] = data['ARRIVAL_DELAY'].apply(func)


# In[19]:


data['result']


# In[20]:


data['result'].sum()


# In[21]:


data.head()


# In[22]:


data['AIR_SYSTEM_DELAY'] = data['AIR_SYSTEM_DELAY'].fillna(data['AIR_SYSTEM_DELAY'].mean())
data['SECURITY_DELAY'] = data['SECURITY_DELAY'].fillna(data['SECURITY_DELAY'].mean())
data['AIRLINE_DELAY'] = data['AIRLINE_DELAY'].fillna(data['AIRLINE_DELAY'].mean())
data['LATE_AIRCRAFT_DELAY'] = data['LATE_AIRCRAFT_DELAY'].fillna(data['LATE_AIRCRAFT_DELAY'].mean())
data['WEATHER_DELAY'] = data['WEATHER_DELAY'].fillna(data['WEATHER_DELAY'].mean())


# In[23]:


data.isnull().sum()


# In[24]:


df = data[[
    'MONTH',
    'DAY',
    'SCHEDULED_DEPARTURE',
    'DEPARTURE_DELAY',
    'SCHEDULED_ARRIVAL',
    'DIVERTED',
    'CANCELLED',
    'AIR_SYSTEM_DELAY',
    'SECURITY_DELAY',
    'AIRLINE_DELAY',
    'LATE_AIRCRAFT_DELAY',
    'WEATHER_DELAY',
    'result'
]]
df.head()


# In[25]:


x = df.drop(['result'], axis = 1)
y = df['result']


# In[26]:


x.head()


# In[27]:


y


# In[28]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)


# In[29]:


scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# In[30]:


clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)


# In[31]:


y_pred = clf.predict(x_test)


# In[32]:


rocAucScore = roc_auc_score(y_test, y_pred)


# In[33]:


rocAucScore


# In[ ]:




