
# coding: utf-8

# # DAT210x - Programming with Python for DS

# ## Module5- Lab3

# In[1]:

import pandas as pd
from datetime import timedelta
import matplotlib.pyplot as plt
import matplotlib

matplotlib.style.use('ggplot') # Look Pretty


# A convenience function for you to use:

# In[2]:

def clusterInfo(model):
    print("Cluster Analysis Inertia: ", model.inertia_)
    print('------------------------------------------')
    
    for i in range(len(model.cluster_centers_)):
        print("\n  Cluster ", i)
        print("    Centroid ", model.cluster_centers_[i])
        print("    #Samples ", (model.labels_==i).sum()) # NumPy Power


# In[3]:

# Find the cluster with the least # attached nodes
def clusterWithFewestSamples(model):
    # Ensure there's at least on cluster...
    minSamples = len(model.labels_)
    minCluster = 0
    
    for i in range(len(model.cluster_centers_)):
        if minSamples > (model.labels_==i).sum():
            minCluster = i
            minSamples = (model.labels_==i).sum()

    print("\n  Cluster With Fewest Samples: ", minCluster)
    return (model.labels_==minCluster)


# ### CDRs

# A [call detail record](https://en.wikipedia.org/wiki/Call_detail_record) (CDR) is a data record produced by a telephone exchange or other telecommunications equipment that documents the details of a telephone call or other telecommunications transaction (e.g., text message) that passes through that facility or device.
# 
# The record contains various attributes of the call, such as time, duration, completion status, source number, and destination number. It is the automated equivalent of the paper toll tickets that were written and timed by operators for long-distance calls in a manual telephone exchange.
# 
# The dataset we've curated for you contains call records for 10 people, tracked over the course of 3 years. Your job in this assignment is to find out where each of these people likely live and where they work at!
# 
# Start by loading up the dataset and taking a peek at its `head` and `dtypes`. You can convert date-strings to real date-time objects using `pd.to_datetime`, and the times using `pd.to_timedelta`:

# In[4]:

# .. your code here ..
df = pd.read_csv(r'E:\Python_SYY\edx\DAT210x\Module5\Datasets\CDR.csv')
df.CallDate = pd.to_datetime(df.CallDate)
df.Duration = pd.to_timedelta(df.Duration)
df.CallTime = pd.to_timedelta(df.CallTime)


# Create a unique list of the phone number values (people) stored in the `In` column of the dataset, and save them in a regular python list called `unique_numbers`. Manually check through `unique_numbers` to ensure the order the numbers appear is the same order they (uniquely) appear in your dataset:

# In[5]:

# .. your code here ..
unique_numbers = df.In.unique().tolist()


# Using some domain expertise, your intuition should direct you to know that people are likely to behave differently on weekends vs on weekdays:
# 
# #### On Weekends
# 1. People probably don't go into work
# 1. They probably sleep in late on Saturday
# 1. They probably run a bunch of random errands, since they couldn't during the week
# 1. They should be home, at least during the very late hours, e.g. 1-4 AM
# 
# #### On Weekdays
# 1. People probably are at work during normal working hours
# 1. They probably are at home in the early morning and during the late night
# 1. They probably spend time commuting between work and home everyday

# In[6]:

print("Examining person: ", 0)


# Create a slice called `user1` that filters to only include dataset records where the `In` feature (user phone number) is equal to the first number on your unique list above:

# In[7]:

# .. your code here ..
user1 = df[df.In == unique_numbers[0]]


# Alter your slice so that it includes only Weekday (Mon-Fri) values:

# In[8]:

user1.dtypes


# In[9]:

# .. your code here ..
user1 = user1[user1.DOW.isin(['Mon','Tue','Wed','Thr','Fri'])]


# The idea is that the call was placed before 5pm. From Midnight-730a, the user is probably sleeping and won't call / wake up to take a call. There should be a brief time in the morning during their commute to work, then they'll spend the entire day at work. So the assumption is that most of the time is spent either at work, or in 2nd, at home:

# In[10]:

# .. your code here ..
user1 = user1[user1.CallTime < '17:00:00']


# Plot the Cell Towers the user connected to

# In[11]:

# .. your code here ..
user1.plot.scatter(x='TowerLon', y='TowerLat', c='gray', alpha=0.1, title='Call Locations')
plt.show()


# In[12]:

def doKMeans(data, num_clusters=0):
    # TODO: Be sure to only feed in Lat and Lon coordinates to the KMeans algo, since none of the other
    # data is suitable for your purposes. Since both Lat and Lon are (approximately) on the same scale,
    # no feature scaling is required. Print out the centroid locations and add them onto your scatter
    # plot. Use a distinguishable marker and color.
    #
    # Hint: Make sure you fit ONLY the coordinates, and in the CORRECT order (lat first). This is part
    # of your domain expertise. Also, *YOU* need to create, initialize (and return) the variable named
    # `model` here, which will be a SKLearn K-Means model for this to work:
    
    # .. your code here ..
    from sklearn.cluster import KMeans
    model = KMeans(n_clusters = num_clusters)
    model.fit(data[['TowerLat','TowerLon']])
    return model


# Let's tun K-Means with `K=3` or `K=4`. There really should only be a two areas of concentration. If you notice multiple areas that are "hot" (multiple areas the user spends a lot of time at that are FAR apart from one another), then increase K=5, with the goal being that all centroids except two will sweep up the annoying outliers and not-home, not-work travel occasions. the other two will zero in on the user's approximate home location and work locations. Or rather the location of the cell tower closest to them.....

# In[32]:

model = doKMeans(user1, 3)


# Print out the mean `CallTime` value for the samples belonging to the cluster with the LEAST samples attached to it. If our logic is correct, the cluster with the MOST samples will be work. The cluster with the 2nd most samples will be home. And the `K=3` cluster with the least samples should be somewhere in between the two. What time, on average, is the user in between home and work, between the midnight and 5pm?

# In[33]:

midWayClusterIndices = clusterWithFewestSamples(model)
midWaySamples = user1[midWayClusterIndices]
print("    Its Waypoint Time: ", midWaySamples.CallTime.mean())


# Let's visualize the results! First draw the X's for the clusters:

# In[34]:

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(model.cluster_centers_[:,1], model.cluster_centers_[:,0], s=169, c='r', marker='x', alpha=0.8, linewidths=2)
ax.set_title('Weekday Calls Centroids')
plt.show()


# In[31]:

model.cluster_centers_.tolist()[1]

