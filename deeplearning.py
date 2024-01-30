#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import yahoo_fin as yf
from yahoo_fin.stock_info import get_data
import matplotlib.pyplot as plt
import scipy
import tensorflow


# In[2]:


np.random.seed(42)
start = '2020-01-13'
end = '2022-01-13'


# In[3]:


meta = get_data('MSFT',start_date=start,end_date=end)
spy = get_data('SPY',start_date=start,end_date=end)


# In[4]:


meta.drop('adjclose',axis=1,inplace=True)


# In[5]:


from sklearn.model_selection import train_test_split


# In[6]:


meta['label1'] = meta['close'].pct_change()


# In[7]:


meta


# In[8]:


meta['spy'] = spy['close']
meta['spy_vol'] = spy['volume']
meta['spy_vol'] /= 100000


# In[9]:


meta['label1'] *= 100


# In[10]:


meta.dropna(inplace=True)


# In[11]:


meta['volume'] = meta['volume']/100000


# In[12]:


meta['label'] = 0


# In[ ]:





# In[13]:


meta['class'], bins = pd.qcut(meta['label1'], q=5, labels=False, retbins=True)


# In[14]:


meta['class'] = meta['class'].astype(int)


# In[15]:


meta['class'] = meta['class'].shift(-1)


# In[16]:


meta.dropna(inplace=True)
meta = meta.reset_index(drop=True)


# In[17]:


one_hot = pd.get_dummies(meta['class'], prefix='class')
meta = pd.concat([meta, one_hot], axis=1)


# In[18]:


meta.head()


# In[19]:


'''meta['class_0.0'] *=2
meta['class_1.0'] *=2
meta['class_2.0'] *=2
meta['class_3.0'] *=2
meta['class_4.0'] *=2''''''


# In[20]:


X_train,X_test,y_train,y_test = train_test_split(meta[['low','high','close','volume','spy','spy_vol']],meta[['class_0.0','class_1.0','class_2.0','class_3.0','class_4.0']],test_size=0.3,random_state=42)


# In[ ]:





# In[21]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers.legacy import Adam


# In[22]:


model = Sequential()


# In[23]:


model.add(Dense(128,input_dim = 6, activation='linear'))
model.add(Dropout(0.5))
model.add(Dense(64,activation='linear'))
model.add(Dropout(0.3))
model.add(Dense(8,activation='linear'))
model.add(Dropout(0.5))
model.add(Dense(5,activation='softmax'))


# In[71]:


model.compile(optimizer = Adam(learning_rate=0.000008), loss='categorical_crossentropy', metrics=['accuracy'])


# In[72]:


model.fit(X_train, y_train,batch_size=64, epochs=2000,validation_split=0.2)


# In[73]:


loss, accuracy = model.evaluate(X_test,y_test)


# In[74]:


meta1 = get_data('MSFT',start_date='2022-01-13',end_date='2024-01-28')


# In[75]:


meta1['label1'] = meta1['close'].pct_change()*100


# In[76]:


meta1.dropna(inplace=True)


# In[77]:


meta1['volume'] = meta1['volume']/100000


# In[78]:


spy1 = get_data('SPY',start_date='2022-01-13',end_date='2024-01-28')
meta1['spy'] = spy1['close']
meta1['spy_vol'] = spy1['volume']/100000


# In[79]:


meta1_pred = meta1[['low','high','close','volume','spy','spy_vol']]


# In[80]:


predicted = model.predict(meta1_pred)


# In[81]:


loss


# In[82]:


predicted


# In[83]:


predicted_class = np.argmax(predicted,axis=1)


# In[86]:


plt.hist(predicted_class)


# In[85]:


plt.plot(actual[300:])


# In[ ]:


actual = np.array(meta1['label1'])


# In[ ]:


predicted[196]


# In[ ]:





# In[206]:


meta[meta['label2']==1]['label1']


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




