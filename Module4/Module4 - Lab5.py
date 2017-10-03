
# coding: utf-8

# # DAT210x - Programming with Python for DS

# ## Module4- Lab5

# In[2]:

import pandas as pd

from scipy import misc
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt


# In[3]:

# Look pretty...

# matplotlib.style.use('ggplot')
plt.style.use('ggplot')


# Create a regular Python list (not NDArray) and name it `samples`:

# In[6]:

# .. your code here ..
samples = []


# Code up a for-loop that iterates over the images in the `Datasets/ALOI/32/` folder. Look in the folder first, so you know how the files are organized, and what file number they start from and end at.
# 
# Load each `.png` file individually in your for-loop using the instructions provided in the Feature Representation reading. Once loaded, flatten the image into a single-dimensional NDArray and append it to your `samples` list.
# 
# **Optional**: You can resample the image down by a factor of two if you have a slower computer. You can also scale the image from `0-255` to `0.0-1.0` if you'd like--doing so shouldn't have any effect on the algorithm's results.

# In[7]:

# .. your code here ..
import os
files_all = os.listdir('E:\Python_SYY\edx\DAT210x\Module4\Datasets\ALOI\\32')
for files in files_all:
    img = misc.imread('E:\Python_SYY\edx\DAT210x\Module4\Datasets\ALOI\\32\\' + files)
    samples.append((img / 255.0).reshape(-1) )
    


# Convert `samples` to a DataFrame named `df`:

# In[9]:

# .. your code here ..
samples = pd.DataFrame(samples)


# Import any necessary libraries to perform Isomap here, reduce `df` down to three components and using `K=6` for your neighborhood size:

# In[32]:

# .. your code here ..
from sklearn import manifold
iso = manifold.Isomap(n_neighbors = 1, n_components = 3)
iso.fit(samples)
T = iso.transform(samples)


# Create a 2D Scatter plot to graph your manifold. You can use either `'o'` or `'.'` as your marker. Graph the first two isomap components:

# In[33]:

# .. your code here ..
T = pd.DataFrame(T)
T.plot.scatter(0,1)
plt.show()


# Chart a 3D Scatter plot to graph your manifold. You can use either `'o'` or `'.'` as your marker:

# In[23]:

# .. your code here ..
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(T.ix[:,0], T.ix[:,2], T.ix[:,2])
plt.show()


# Answer the first three lab questions!

# Create another for loop. This time it should iterate over all the images in the `Datasets/ALOI/32_i` directory. Just like last time, load up each image, process them the way you did previously, and append them into your existing `samples` list:

# In[35]:

# .. your code here ..files_all = os.listdir('E:\Python_SYY\edx\DAT210x\Module4\Datasets\ALOI\\32')
samples = []
colors = []
files_all = os.listdir('E:\Python_SYY\edx\DAT210x\Module4\Datasets\ALOI\\32')
for files in files_all:
    img = misc.imread('E:\Python_SYY\edx\DAT210x\Module4\Datasets\ALOI\\32\\' + files)
    samples.append((img / 255.0).reshape(-1) )
    colors.append('b')
files_all = os.listdir('E:\Python_SYY\edx\DAT210x\Module4\Datasets\ALOI\\32i')
for files in files_all:
    img = misc.imread('E:\Python_SYY\edx\DAT210x\Module4\Datasets\ALOI\\32i\\' + files)
    samples.append((img / 255.0).reshape(-1) )
    colors.append('r')


# Convert `samples` to a DataFrame named `df`:

# In[36]:

# .. your code here ..
df = pd.DataFrame(samples)


# Import any necessary libraries to perform Isomap here, reduce `df` down to three components and using `K=6` for your neighborhood size:

# In[37]:

# .. your code here ..
from sklearn import manifold
iso = manifold.Isomap(n_neighbors = 6, n_components = 3)
iso.fit(df)
T = iso.transform(df)


# Create a 2D Scatter plot to graph your manifold. You can use either `'o'` or `'.'` as your marker. Graph the first two isomap components:

# In[38]:

# .. your code here ..
T = pd.DataFrame(T)
T.plot.scatter(0,1,color)
plt.show()


# Chart a 3D Scatter plot to graph your manifold. You can use either `'o'` or `'.'` as your marker:

# In[39]:

# .. your code here ..
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(T.ix[:,0], T.ix[:,2], T.ix[:,2])
plt.show()


# In[ ]:



