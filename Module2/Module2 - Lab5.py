
# coding: utf-8

# # DAT210x - Programming with Python for DS

# ## Module2 - Lab5

# Import and alias Pandas:

# In[1]:

# .. your code here ..
import pandas as pd


# As per usual, load up the specified dataset, setting appropriate header labels.

# In[2]:

# .. your code here ..
df = pd.read_csv('E:\Python_SYY\edx\DAT210x\Module2\Datasets\census.data',header = None)
df = df.drop(0,axis = 1)
df.reset_index(drop = True)
df.columns = ['education', 'age', 'capital-gain', 'race', 'capital-loss', 'hours-per-week', 'sex', 'classification']


# Excellent.
# 
# Now, use basic pandas commands to look through the dataset. Get a feel for it before proceeding!
# 
# Do the data-types of each column reflect the values you see when you look through the data using a text editor / spread sheet program? If you see `object` where you expect to see `int32` or `float64`, that is a good indicator that there might be a string or missing value or erroneous value in the column.

# In[3]:

# .. your code here ..
df['capital-gain'] = pd.to_numeric(df['capital-gain'],errors = 'coerce')


# Try use `your_data_frame['your_column'].unique()` or equally, `your_data_frame.your_column.unique()` to see the unique values of each column and identify the rogue values.
# 
# If you find any value that should be properly encoded to NaNs, you can convert them either using the `na_values` parameter when loading the dataframe. Or alternatively, use one of the other methods discussed in the reading.

# In[4]:

# .. your code here ..
#见上一个框


# Look through your data and identify any potential categorical features. Ensure you properly encode any ordinal and nominal types using the methods discussed in the chapter.
# 
# Be careful! Some features can be represented as either categorical or continuous (numerical). If you ever get confused, think to yourself what makes more sense generally---to represent such features with a continuous numeric type... or a series of categories?

# In[5]:

# .. your code here ..
df.education.unique()
ordered_education = ['Preschool','1st-4th', '5th-6th','7th-8th','9th','10th','11th','12th','HS-grad','Bachelors','Doctorate']
df.education = df.education.astype("category",ordered=True,categories=ordered_education).cat.codes
ordered_classification = ['<=50K','>50K']
df.classification = df.classification.astype("category",ordered=True,categories=ordered_classification).cat.codes
df2 = pd.get_dummies(df,columns=['race','sex'])


# Lastly, print out your dataframe!

# In[7]:

# .. your code here ..
df2

