
# coding: utf-8

# # DAT210x - Programming with Python for DS

# ## Module3 - Lab6

# In[1]:

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib


# In[2]:

# Look pretty...

# matplotlib.style.use('ggplot')
plt.style.use('ggplot')


# Load up the wheat seeds dataset into a dataframe. We've stored a copy in the Datasets directory.

# In[3]:

# .. your code here ..
df = pd.read_csv('E:\Python_SYY\edx\DAT210x\Module3\Datasets\wheat.data')


# If you loaded the `id` column as a feature (hint: _you shouldn't have!_), then be sure to drop it. Also get rid of the `area` and `perimeter` features:

# In[4]:

# .. your code here ..
df2 = df.drop(['id'],axis = 1)


# Compute the correlation matrix of your dataframe:

# In[5]:

# .. your code here ..
df2.corr()


# Graph the correlation matrix using `imshow` or `matshow`:

# In[9]:

# .. your code here ..
plt.imshow(df2.corr(), cmap=plt.cm.Blues, interpolation='nearest')
plt.colorbar()
tick_marks = [i for i in range(len(df2.columns))]
plt.xticks(tick_marks, df2.columns, rotation='vertical')
plt.yticks(tick_marks, df2.columns)
plt.show()


# In[8]:

# Display the graphs:
plt.show()


# In[ ]:



