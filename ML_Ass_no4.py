#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from io import IncrementalNewlineDecoder
import numpy as np #Linear algebra
import pandas as pd #Data processing, csv file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #for data visualization
import seaborn as sns #for statistical data visualization
get_ipython().run_line_magic('matplotlib', 'inline')
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
  for filename in filenames:
    print(os.path.join(dirname, filename))


# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# In[ ]:


df=pd.read_csv("sales_data_sample.csv",sep=",",encoding='Latin-1')


# In[ ]:


df.shape


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


df.isnull().sum()


# In[ ]:


df.drop(['Column1','Column2','Column3','Column4'],axis=1,inplace=True)


# In[ ]:


df.info()


# In[ ]:


df.describe()


# In[ ]:


#View the labels in the variable

df['ORDERNUMBER'].unique()


# In[ ]:


#View how many different types of variable are there
len(df['ORDERNUMBER'].unique())


# In[ ]:


#View the labels in the variable
df['QUANTITYORDERED'].unique()


# In[ ]:


len(df['QUANTITYORDERED'].unique())


# In[ ]:


df['SALES'].unique()


# In[ ]:


len(df['SALES'].unique())


# In[ ]:


df.drop(['ORDERNUMBER','QUANTITYORDERED'],axis=1,inplace=True)


# In[ ]:


df.info()


# In[ ]:


df.head()


# In[ ]:


x=df
y=df['SALES']


# In[ ]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
x['status_type']=le.fit_transform(x['SALES'])
y=le.transform(y)


# In[ ]:


x.info()


# In[ ]:


x.head()


# In[ ]:


cols=x.columns


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
ms=MinMaxScaler()
x=ms.fit_transform(x)


# In[ ]:


x=pd.DataFrame(x,columns=[cols])


# In[ ]:


x.head()


# In[ ]:


from sklearn.cluster import KMeans
kmeans=KMeans(n_clusters=2,random_state=0)
kmeans.fit(x)


# In[ ]:


kmeans.cluster_centers()


# In[ ]:


kmeans.inertia_


# In[ ]:


labels=kmeans.labels_
#Check how many of the samples were correctly labeled
correct_labels=sum(y==labels)
print("Result:%d out of %d samples were correctly labeled."%(correct_labels,y.size))


# In[ ]:


print('Accuracy score:{0:0.2f}'.format(correct_labels/float(y.size)))


# In[ ]:


from sklearn.cluster import KMeans
cs=[]
for i in range(1,11):
  kmeans=KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
  kmeans.fit(x)
  cs.append(kmeans.inertia_)
plt.plot(range(1,11),cs)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('cs')
plt.show()


# In[ ]:


from sklearn.cluster import KMeans
kmeans=KMeans(n_clusters=2,random_state=0)
kmeans.fit(x)
labels=kmeans.labels_
#Check how many of the samples were correctly labeled
correct_labels=sum(y==lables)
print("Result: %d out of %d samples were correctly labeled."%(correct_labels,y.size))
print('Accuracy score: {0:0.2f}'.format(correct_labels/float(y.size)))


# In[ ]:


kmeans=KMeans(n_clusters=3,random_state=0)
kmeans.fit(x)
labels=kmeans.labels_
#Check how many of the samples were correctly labeled
correct_labels=sum(y==lables)
print("Result: %d out of %d samples were correctly labeled."%(correct_labels,y.size))
print('Accuracy score: {0:0.2f}'.format(correct_labels/float(y.size)))


# In[ ]:


kmeans=KMeans(n_clusters=4,random_state=0)
kmeans.fit(x)
labels=kmeans.labels_
#Check how many of the samples were correctly labeled
correct_labels=sum(y==lables)
print("Result: %d out of %d samples were correctly labeled."%(correct_labels,y.size))
print('Accuracy score: {0:0.2f}'.format(correct_labels/float(y.size)))

