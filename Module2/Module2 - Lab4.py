
# coding: utf-8

# # DAT210x - Programming with Python for DS

# ## Module2 - Lab4

# Import and alias Pandas:

# In[1]:

# .. your code here ..
import pandas as pd


# Load up the table from the link, and extract the dataset out of it. If you're having issues with this, look carefully at the sample code provided in the reading:

# In[2]:

# .. your code here ..
df = pd.read_html('http://www.espn.com/nhl/statistics/player/_/stat/points/sort/points/year/2015/seasontype/2`')[0]


# Next up, rename the columns so that they are _similar_ to the column definitions provided to you on the website. Be careful and don't accidentally use any column names twice. If a column uses special characters, you can replace them with regular characters to make it easier to work with:

# In[3]:

# .. your code here .
c_name = df.iloc[1].tolist()
c_name[13:17] = ['PP_G','PP_A','SH_G','SH_A']
df.columns = c_name
df.drop([0,1],inplace = True)
df.reset_index(drop = True)


# Get rid of any row that has at least 4 NANs in it. That is, any rows that do not contain player points statistics:

# In[5]:

# .. your code here ..
df = df.dropna(axis=0, thresh = 4)
df


# At this point, look through your dataset by printing it. There probably still are some erroneous rows in there. What indexing command(s) will you use to select all rows EXCEPT those rows?

# In[8]:

# .. your code here ..
df2 = df.loc[df.RK != "RK"]


# Get rid of the 'RK' column:

# In[12]:

# .. your code here ..
df2.drop('RK', axis=1, inplace=True)


# Make sure there are no holes in your index by resetting it. There is an example of this in the reading material. By the way, drop the original index.

# In[13]:

# .. your code here ..
df2.reset_index(drop=True)


# Check the data type of all columns, and ensure those that should be numeric are numeric.

# In[51]:

# .. your code here.
lst = df2.columns.tolist()
lst = lst[2:]
for x in lst:
   pd.to_numeric(df2[x],errors='coerce')


# Your dataframe is now ready! Use the appropriate commands to answer the questions on the course lab page.

# In[61]:

# .. your code here ..
df2.reset_index(drop=True)
len(df2.PCT.unique())
df2.reset_index(drop=True)

