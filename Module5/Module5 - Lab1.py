
# coding: utf-8

# # DAT210x - Programming with Python for DS

# ## Module5- Lab1

# Start by importing whatever you need to import in order to make this lab work:

# In[1]:

# .. your code here ..
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# ### How to Get The Dataset

# 1. Open up the City of Chicago's [Open Data | Crimes](https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-present/ijzp-q8t2) page.
# 1. In the `Primary Type` column, click on the `Menu` button next to the info button, and select `Filter This Column`. It might take a second for the filter option to show up, since it has to load the entire list first.
# 1. Scroll down to `GAMBLING`
# 1. Click the light blue `Export` button next to the `Filter` button, and select `Download As CSV`

# Now that you have th dataset stored as a CSV, load it up being careful to double check headers, as per usual:

# In[2]:

# .. your code here ..
df = pd.read_csv(r'E:\Python_SYY\edx\DAT210x\Module5\Datasets\Crimes_-_2001_to_present.csv')


# Get rid of any _rows_ that have nans in them:

# In[3]:

# .. your code here ..
df = df.dropna(axis = 0)


# Display the `dtypes` of your dset:

# In[4]:

# .. your code here ..
df.dtypes


# Coerce the `Date` feature (which is currently a string object) into real date, and confirm by displaying the `dtypes` again. This might be a slow executing process...

# In[5]:

# .. your code here ..
df.Date = pd.to_datetime(df.Date)
df.dtypes


# In[10]:

def doKMeans(df):
    # Let's plot your data with a '.' marker, a 0.3 alpha at the Longitude,
    # and Latitude locations in your dataset. Longitude = x, Latitude = y
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(df.Longitude, df.Latitude, marker='.', alpha=0.3)

    
    # TODO: Filter `df` using indexing so it only contains Longitude and Latitude,
    # since the remaining columns aren't really applicable for this lab:
    #
    # .. your code here ..
    df2 = df[['Longitude', 'Latitude']]


    # TODO: Use K-Means to try and find seven cluster centers in this df.
    # Be sure to name your kmeans model `model` so that the printing works.
    #
    # .. your code here ..
    model = KMeans(n_clusters = 7)
    model.fit(df2)
    
    # Now we can print and plot the centroids:
    centroids = model.cluster_centers_
    print(centroids)
    ax.scatter(centroids[:,0], centroids[:,1], marker='x', c='red', alpha=0.5, linewidths=3, s=169)
    plt.show()


# In[23]:

# Print & Plot your data
doKMeans(df)


# Filter out the data so that it only contains samples that have a `Date > '2011-01-01'`, using indexing. Then, in a new figure, plot the crime incidents, as well as a new K-Means run's centroids.

# In[13]:

# .. your code here ..
df_new = df[df.Date > '2011-01-01']


# In[20]:

# Print & Plot your data
doKMeans(df_new)


# In[ ]:



