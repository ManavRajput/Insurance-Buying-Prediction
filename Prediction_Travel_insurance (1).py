#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


df = pd.read_csv("TravelInsurancePrediction.csv")
df


# In[2]:


from sklearn.model_selection import train_test_split


features = ['Age','AnnualIncome','FamilyMembers','ChronicDiseases','FrequentFlyer','EverTravelledAbroad','GraduateOrNot']
x = df.loc[:, features]
y = df.loc[:, ['TravelInsurance']]

X_train,X_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 25)


# In[ ]:





# In[ ]:





# In[3]:


X_train_scaled = X_train.copy()
X_train_scaled['AnnualIncome'] = X_train_scaled['AnnualIncome'] / 1000000
X_train_scaled['Age'] = X_train_scaled['Age'] / 100
X_train_scaled['FamilyMembers'] = X_train_scaled['FamilyMembers'] / 10
X_train_scaled['FrequentFlyer'] = X_train_scaled['FrequentFlyer'].map({'Yes': 1, 'No': 0})
X_train_scaled['EverTravelledAbroad'] = X_train_scaled['EverTravelledAbroad'].map({'Yes': 1, 'No': 0})
X_train_scaled['GraduateOrNot'] = X_train_scaled['GraduateOrNot'].map({'Yes': 1, 'No': 0})

print(X_train_scaled[:10])

X_test_scaled = X_test.copy()
X_test_scaled['Age'] = X_test_scaled['Age'] / 100
X_test_scaled['AnnualIncome'] = X_test_scaled['AnnualIncome'] / 10000
X_test_scaled['FamilyMembers'] = X_test_scaled['FamilyMembers'] / 10
X_test_scaled['FrequentFlyer'] = X_test_scaled['FrequentFlyer'].map({'Yes': 1, 'No': 0})
X_test_scaled['EverTravelledAbroad'] = X_test_scaled['EverTravelledAbroad'].map({'Yes': 1, 'No': 0})
X_test_scaled['GraduateOrNot'] = X_test_scaled['GraduateOrNot'].map({'Yes': 1, 'No': 0})


# In[32]:


model = keras.Sequential([
    keras.layers.Dense(100,input_shape = (7,) , activation = 'relu'),
    keras.layers.Dense(50,activation = 'sigmoid'),
    keras.layers.Dense(1,activation = 'sigmoid')
])

# model.compile(optimizer = 'adam',
#              loss = 'MeanAbsoluteError',
#              metrics = ['accuracy']
#              )


# In[ ]:





# In[33]:


model.compile(optimizer = 'adam',
             loss = 'mean_squared_logarithmic_error',
             metrics = ['accuracy']
             )


# In[34]:


model.fit(X_train_scaled, y_train, epochs = 50)


# In[ ]:





# In[ ]:





# In[39]:


for x in model.predict(X_test_scaled):
    if (x > 0.5):
        print("0")
    else:
        print("1")


# In[42]:


# # import everything from tkinter module
from tkinter import * 
from tkinter import simpledialog
from tkinter import messagebox

# create a tkinter window
root = Tk()             
 
    
root.withdraw()
sum = 'Y'
while(sum!='N'):
    # the input dialog
    a = simpledialog.askstring(title="Travel",
                                      prompt="Age : ")
    b = simpledialog.askstring(title="Travel",
                                      prompt="AnnualIncome($): ")
    c = simpledialog.askstring(title="Travel",
                                      prompt="FamilyMembers : ")
    d = simpledialog.askstring(title="Travel",
                                      prompt="ChronicDiseases (1/0) : ")

    e = simpledialog.askstring(title="Travel",
                                      prompt="EverTravelledAbroad(1/0) : ")
    f = simpledialog.askstring(title="Travel",
                                      prompt="FrequentFlyer (1/0) : ")
    g = simpledialog.askstring(title="Travel",
                                      prompt="GraduateOrNot(1/0) : ")
    
    
    
    
    def fun(num): 
        if num > 0.7:
            messagebox.showinfo("Will buy the Insurrance")  
        else:
            messagebox.showinfo("Will not buy the Insurrance") 
    root.geometry('500x600') 


     # Set the position of button on the top of window.

    # btn.pack(side = 'bottom')     



    converted_num_a = int(a)
    converted_num_b = int(b)
    converted_num_c = int(c)
    converted_num_d = int(d)
    converted_num_e = int(e)
    converted_num_f = int(f)
    converted_num_g = int(g)


    data = {
        'Age' : [converted_num_a%100] ,
        'AnnualIncome': [converted_num_b] ,
        'FamilyMembers' : [converted_num_c] ,
        'ChronicDiseases' : [converted_num_d] ,
        'EverTravelledAbroad' : [converted_num_e] ,
        'FrequentFlyer' : [converted_num_f] ,
        'GraduateOrNot' : [converted_num_g] ,
    }

    dff = pd.DataFrame(data)
    n = model.predict(dff)

    btn2 = Button(root, text = 'Hello', bd = '5',command = fun(n))
    btn2.pack(side = 'top')
    
    num = simpledialog.askstring(title="Test",
                                      prompt="Do you want to continue : (Y/N)")
    
    sum = num


# In[ ]:




