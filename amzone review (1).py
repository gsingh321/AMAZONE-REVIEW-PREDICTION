#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as nm
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer


# In[2]:


df = pd.read_csv(r'C:\Users\dell\Downloads\New folder (2)\Reviews.csv')


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.info()


# In[5]:


df1=df.iloc[:,4:]


# In[6]:


df1


# In[7]:


df1=df1.drop('Time',axis=1)


# In[8]:


df1.isnull().sum()           # to check null values in the columns


# In[9]:


df1[df1.Summary.isnull()]       


# In[10]:


df1=df1.dropna()  # droping nan values


# In[11]:


df1


# In[12]:


rev=df1.Score.value_counts(normalize=True)


# In[13]:


plt.figure(figsize=(5,5))
plt.pie(rev,labels=rev.keys(),autopct='%0.1f%%')          # check the percentage of different feedbacks star given by customers
plt.title('PERCENTAGE OF REVIEW SCORE')
plt.show()


# In[14]:


score=[]
for i in df1.Score:
    if i==5:
        score.append(1)                    # replace 5 and 4 postive review with 1 and 1,2 with negative review
    elif i==4:
        score.append(1)
    elif i==3:
        score.append('N')
    elif i==2:
        score.append(0)
    elif i==1:
        score.append(0)
df1['Score']=score


# In[15]:


df1=df1[df1.Score!='N']


# In[16]:


df1.head()


# In[17]:


# cleaning the text column

df1.Text=df1.Text.replace(r'<br .>',' ',regex=True)   


# In[18]:


df1


# In[19]:


df1.Text=df1.Text.replace('[^a-zA-Z]',' ',regex=True)


# In[20]:


stopwords = set(stopwords.words('english'))


# In[21]:


lem =WordNetLemmatizer()


# In[22]:


# define funtion for replaceing word to root word and removing the stopword in the columns

def rem(x):
     return ' '.join(lem.lemmatize(word) for word in str(x).split() if  word not in stopwords)


# In[23]:


df1['Text']=df1['Text'].apply(lambda x:rem(x))


# In[24]:


df1


# In[25]:


cv=CountVectorizer(max_features=41157,ngram_range=(1,2),min_df=10)


# In[26]:


df1.Text=df1.Text.str.lower()


# In[27]:


df1.Summary=df1.Summary.str.lower()            # for for converting all the text in the lower case 


# In[28]:


df1


# In[29]:


X=cv.fit_transform(df1.Text)          
y=df1.Score


# In[30]:


y=y.astype('int')


# In[31]:


x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3)               #splitting data into two parts train and test


# In[32]:


lr=LogisticRegression()


# In[33]:


lr.fit(x_train,y_train)


# In[34]:


y_pred=lr.predict(x_test)


# In[35]:


cm=confusion_matrix(y_pred,y_test)


# In[36]:


cm


# In[37]:


accuracy=(cm[0][0] +cm[1][1])/(cm[0][0] +cm[0][1]+cm[1][0] +cm[1][1])


# In[38]:


sensitivity = cm[1,1]/(cm[1,0,]+cm[1,1])
specificity =cm[0,0]/(cm[0,0]+cm[0,1])


# In[39]:


print('accuracy of the model:',accuracy)
print('sesitivity of the model:',sensitivity)
print('specificity of the model:',specificity)


# In[40]:


y_pred_proba = lr.predict_proba(x_test)[::,1]


# In[41]:


fpr, tpr,_ = metrics.roc_curve(y_test,  y_pred_proba)


# In[42]:


y_pred_proba = lr.predict_proba(x_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)

plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()


# In[ ]:





# In[ ]:




