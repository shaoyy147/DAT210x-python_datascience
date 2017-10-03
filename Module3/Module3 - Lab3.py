
# coding: utf-8

# # DAT210x - Programming with Python for DS

# ## Module3 - Lab3

# In[1]:

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib


# In[2]:

# Look pretty...

# matplotlib.style.use('ggplot')
plt.style.use('ggplot')


# Load up the wheat seeds dataset into a dataframe. We've stored a copy in the Datasets directory.

# In[5]:

# .. your code here ..
df = pd.read_csv('E:\Python_SYY\edx\DAT210x\Module3\Datasets\wheat.data',header = 0)


# Create a new 3D subplot using figure `fig`, which we've defined for you below. Use that subplot to draw a 3D scatter plot using the `area`, `perimeter`, and `asymmetry` features. Be sure so use the optional display parameter `c='red'`, and also label your axes:

# In[7]:

fig = plt.figure()

# .. your code here ..
from mpl_toolkits.mplot3d import Axes3D
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('area')
ax.set_ylabel('perimeter')
ax.set_zlabel('asymmetry')
ax.scatter(df.area, df.perimeter, df.asymmetry, c='r', marker='.')
plt.show()


# Create another 3D subplot using fig. Then use the subplot to graph a 3D scatter plot of the `width`, `groove`, and `length` features. Be sure so use the optional display parameter `c='green'`, and be sure to label your axes:

# In[9]:

fig = plt.figure()

# .. your code here ..
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('width')
ax.set_ylabel('groove')
ax.set_zlabel('length')
ax.scatter(df.width, df.groove, df.length, c='green', marker='.')
plt.show()


# Create a 2d scatter plot that graphs the `compactness` and `width` features:

# In[8]:

# Display the graphs:
df.plot.scatter(x='compactness',y='width')
plt.show()


# In[ ]:



