
# coding: utf-8

# # DAT210x - Programming with Python for DS

# ## Module3 - Lab4

# In[1]:

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# This is new
from pandas.tools.plotting import parallel_coordinates


# In[2]:

# Look pretty...

# matplotlib.style.use('ggplot')
plt.style.use('ggplot')


# Load up the wheat seeds dataset into a dataframe. We've stored a copy in the Datasets directory.

# In[3]:

# .. your code here ..
df = pd.read_csv('E:\Python_SYY\edx\DAT210x\Module3\Datasets\wheat.data',header = 0)


# If you loaded the `id` column as a feature (hint: _you shouldn't have!_), then be sure to drop it. Also get rid of the `area` and `perimeter` features:

# In[10]:

# .. your code here ..
df2 = df.drop(['id','area','perimeter'],axis=1)


# Plot a parallel coordinates chart grouped by the `wheat_type` feature. Set the optional display parameter `alpha` to `0.4`:

# In[12]:

# .. your code here ..
plt.figure()
parallel_coordinates(df2, 'wheat_type')


# Create a 2d scatter plot that graphs the `compactness` and `width` features:

# In[13]:

# Display the graphs:
df2.plot.scatter(x='compactness',y='width')
plt.show()


# In[ ]:



