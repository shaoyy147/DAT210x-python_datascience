
# coding: utf-8

# # DAT210x - Programming with Python for DS

# ## Module2 - Lab2

# In[1]:

# Import and alias Pandas
import pandas as pd


# Write code below to load up the `tutorial.csv` dataset. You can store it into a variable called df:

# In[4]:

# .. your code here ..
df = pd.read_csv(r'E:\Python_SYY\edx\DAT210x\Module2\Datasets\tutorial.csv')


# Now that your dataset has been loaded, invoke the `.describe()` method to display some results about it:

# In[5]:

# .. your code here ..
df.describe()


# Another _very_ useful method you can use to get an overview of your data is by using the `.summary()` method. This returns even more details than `.describe()`. Try it out below:

# In[10]:

# .. your code here ..
df.summary()


# Lastly, try experimenting with indexing. Figure out which _indexing method_ you would need to use in order to index your dataframe with: `[2:4, 'col3']`. Finally, display the results:

# In[9]:

# .. your code here ..
df.ix[2:4,'col3']

