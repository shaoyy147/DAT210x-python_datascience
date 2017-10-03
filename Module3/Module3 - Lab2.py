
# coding: utf-8

# # DAT210x - Programming with Python for DS

# ## Module3 - Lab2

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
df = pd.read_csv('E:\Python_SYY\edx\DAT210x\Module3\Datasets\wheat.data',header = 0)


# Create a 2d scatter plot that graphs the `area` and `perimeter` features:

# In[4]:

# .. your code here ..
df.plot.scatter(x='area',y='perimeter')
plt.show()


# Create a 2d scatter plot that graphs the `groove` and `asymmetry` features:

# In[5]:

# .. your code here ..
df.plot.scatter(x='groove',y='asymmetry')
plt.show()


# Create a 2d scatter plot that graphs the `compactness` and `width` features:

# In[6]:

# .. your code here ..
df.plot.scatter(x='compactness',y='width')
plt.show()


# ### BONUS
# 
# After completing the above, go ahead and run your program Check out the results, and see what happens when you add in the optional display parameter marker with values of either `'^'`, `'.'`, or `'o'`:

# In[7]:

# .. your code here ..
df.plot.scatter(x='groove',y='asymmetry',marker='.')


# In[8]:

# Display the graphs:
plt.show()


# In[ ]:



