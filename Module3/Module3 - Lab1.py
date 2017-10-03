
# coding: utf-8

# # DAT210x - Programming with Python for DS

# ## Module3 - Lab1

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
df = pd.read_csv('E:\Python_SYY\edx\DAT210x\Module3\Datasets\wheat.data', header = 0)


# Create a slice from your dataframe and name the variable `s1`. It should only include the `area` and `perimeter` features.

# In[8]:

# .. your code here ..
s1 = df[['area','perimeter']]


# Create another slice of from dataframe called it `s2` this time. Slice out only the `groove` and `asymmetry` features:

# In[9]:

# .. your code here ..
s2 = df[['groove','asymmetry']]


# Create a histogram plot using the first slice, and another histogram plot using the second slice. Be sure to set `alpha=0.75`.

# In[10]:

# .. your code here ..
s1.plot.hist(alpha = 0.75)
s2.plot.hist(alpha = 0.75)


# In[11]:

# Display the graphs:
plt.show()


# In[ ]:



