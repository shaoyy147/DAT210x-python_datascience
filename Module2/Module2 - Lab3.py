
# coding: utf-8

# # DAT210x - Programming with Python for DS

# ## Module2 - Lab3

# In[1]:

# Import and alias Pandas
import pandas as pd


# Often, you will want to load a dataset that is missing explicit header labels. You won't know if your data lacks headers or not unless you load it up and examine the headers to see if they make sense. Pandas by default reads in the first row of data as the header. If that isn't the case for your specific data set, you will lose your first data row. Be careful!
# 
# Load up the `Servo.data` dataset. Examine the headers, and adjust them as necessary, if need be.

# In[5]:

# .. your code here ..
df = pd.read_csv(r'E:\Python_SYY\edx\DAT210x\Module2\Datasets\Servo.data',header = None, names=['motor', 'screw', 'pgain', 'vgain', 'class'])
df.head(5)


# Let's try experimenting with some slicing. Create a slice that contains all entries that have a vgain equal to 5. Then print the length of (# of samples in) that slice:

# In[8]:

# .. your code here ..
len(df[df.vgain == 5])


# Create a slice that contains all entries having a motor equal to E and screw equal to E. Then print the length of (# of samples in) that slice:

# In[15]:

# .. your code here ..
len(df[(df.motor == 'E') & (df.screw == 'E')])


# Create a slice that contains all entries having a pgain equal to 4. Use one of the various methods of finding the mean vgain value for the samples in that slice. Once you've found it, print it:

# In[11]:

# .. your code here ..
df[df.pgain == 4].describe()


# Here's a bonus activity for you. See what happens when you display the `.dtypes` property of your dataframe!

# In[12]:

# .. your code here ..
df.dtypes

