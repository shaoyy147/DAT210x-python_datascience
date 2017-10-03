
# coding: utf-8

# # DAT210x - Programming with Python for DS

# ## Module4- Lab2

# In[1]:

import math
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

from sklearn import preprocessing


# In[2]:

# Look pretty...

# matplotlib.style.use('ggplot')
plt.style.use('ggplot')


# ### Some Boilerplate Code

# For your convenience, we've included some boilerplate code here which will help you out. You aren't expected to know how to write this code on your own at this point, but it'll assist with your visualizations. We've added some notes to the code in case you're interested in knowing what it's doing:

# ### A Note on SKLearn's `.transform()` calls:

# Any time you perform a transformation on your data, you lose the column header names because the output of SciKit-Learn's `.transform()` method is an NDArray and not a daraframe.
# 
# This actually makes a lot of sense because there are essentially two types of transformations:
# - Those that adjust the scale of your features, and
# - Those that change alter the number of features, perhaps even changing their values entirely.
# 
# An example of adjusting the scale of a feature would be changing centimeters to inches. Changing the feature entirely would be like using PCA to reduce 300 columns to 30. In either case, the original column's units have either been altered or no longer exist at all, so it's up to you to assign names to your columns after any transformation, if you'd like to store the resulting NDArray back into a dataframe.

# In[3]:

def scaleFeaturesDF(df):
    # Feature scaling is a type of transformation that only changes the
    # scale, but not number of features. Because of this, we can still
    # use the original dataset's column names... so long as we keep in
    # mind that the _units_ have been altered:

    scaled = preprocessing.StandardScaler().fit_transform(df)
    scaled = pd.DataFrame(scaled, columns=df.columns)
    
    print("New Variances:\n", scaled.var())
    print("New Describe:\n", scaled.describe())
    return scaled


# SKLearn contains many methods for transforming your features by scaling them, a type of pre-processing):
#     - `RobustScaler`
#     - `Normalizer`
#     - `MinMaxScaler`
#     - `MaxAbsScaler`
#     - `StandardScaler`
#     - ...
# 
# http://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing
# 
# However in order to be effective at PCA, there are a few requirements that must be met, and which will drive the selection of your scaler. PCA requires your data is standardized -- in other words, it's _mean_ should equal 0, and it should have unit variance.
# 
# SKLearn's regular `Normalizer()` doesn't zero out the mean of your data, it only clamps it, so it could be inappropriate to use depending on your data. `MinMaxScaler` and `MaxAbsScaler` both fail to set a unit variance, so you won't be using them here either. `RobustScaler` can work, again depending on your data (watch for outliers!). So for this assignment, you're going to use the `StandardScaler`. Get familiar with it by visiting these two websites:
# 
# - http://scikit-learn.org/stable/modules/preprocessing.html#preprocessing-scaler
# - http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler

# Lastly, some code to help with visualizations:

# In[4]:

def drawVectors(transformed_features, components_, columns, plt, scaled):
    if not scaled:
        return plt.axes() # No cheating ;-)

    num_columns = len(columns)

    # This funtion will project your *original* feature (columns)
    # onto your principal component feature-space, so that you can
    # visualize how "important" each one was in the
    # multi-dimensional scaling

    # Scale the principal components by the max value in
    # the transformed set belonging to that component
    xvector = components_[0] * max(transformed_features[:,0])
    yvector = components_[1] * max(transformed_features[:,1])

    ## visualize projections

    # Sort each column by it's length. These are your *original*
    # columns, not the principal components.
    important_features = { columns[i] : math.sqrt(xvector[i]**2 + yvector[i]**2) for i in range(num_columns) }
    important_features = sorted(zip(important_features.values(), important_features.keys()), reverse=True)
    print("Features by importance:\n", important_features)

    ax = plt.axes()

    for i in range(num_columns):
        # Use an arrow to project each original feature as a
        # labeled vector on your principal component axes
        plt.arrow(0, 0, xvector[i], yvector[i], color='b', width=0.0005, head_width=0.02, alpha=0.75)
        plt.text(xvector[i]*1.2, yvector[i]*1.2, list(columns)[i], color='b', alpha=0.75)

    return ax


# ### And Now, The Assignment

# In[5]:

# Do * NOT * alter this line, until instructed!
scaleFeatures = True


# Load up the dataset specified on the lab instructions page and remove any and all _rows_ that have a NaN in them. You should be a pro at this by now ;-)
# 
# **QUESTION**: Should the `id` column be included in your dataset as a feature?

# In[6]:

# .. your code here ..
df = pd.read_csv('E:\Python_SYY\edx\DAT210x\Module4\Datasets\kidney_disease.csv')
df.dropna(axis =0, inplace = True)
df.drop('id',axis = 1,inplace = True)
df.reset_index(inplace = True, drop = True)
df


# Let's build some color-coded labels; the actual label feature will be removed prior to executing PCA, since it's unsupervised. You're only labeling by color so you can see the effects of PCA:

# In[7]:

labels = ['red' if i=='ckd' else 'green' for i in df.classification]


# Use an indexer to select only the following columns: `['bgr','wc','rc']`

# In[8]:

# .. your code here ..
df2 = df[['bgr','wc','rc']]
df2.head()
df2.dtypes


# Either take a look at the dataset's webpage in the attribute info section of UCI's [Chronic Kidney Disease]() page,: https://archive.ics.uci.edu/ml/datasets/Chronic_Kidney_Disease or alternatively, you can actually look at the first few rows of your dataframe using `.head()`. What kind of data type should these three columns be? Compare what you see with the results when you print out your dataframe's `dtypes`.
# 
# If Pandas did not properly detect and convert your columns to the data types you expected, use an appropriate command to coerce these features to the right type.

# In[9]:

# .. your code here ..
df2.bgr = pd.to_numeric(df2.bgr)
df2.wc = pd.to_numeric(df2.wc)
df2.rc = pd.to_numeric(df2.rc)


# PCA Operates based on variance. The variable with the greatest variance will dominate. Examine your data using a command that will check the variance of every feature in your dataset, and then print out the results. Also print out the results of running `.describe` on your dataset.
# 
# _Hint:_ If you do not see all three variables: `'bgr'`, `'wc'`, and `'rc'`, then it's likely you probably did not complete the previous step properly.

# In[10]:

# .. your code here ..
df2.describe()


# Below, we assume your dataframe's variable is named `df`. If it isn't, make the appropriate changes. But do not alter the code in `scaleFeaturesDF()` just yet!

# In[11]:

# .. your (possible) code adjustment here ..
df = df2
if scaleFeatures: df = scaleFeaturesDF(df)


# Run PCA on your dataset, reducing it to 2 principal components. Make sure your PCA model is saved in a variable called `'pca'`, and that the results of your transformation are saved in another variable `'T'`:

# In[12]:

# .. your code here ..
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
pca.fit(df)
T = pca.transform(df)    


# Now, plot the transformed data as a scatter plot. Recall that transforming the data will result in a NumPy NDArray. You can either use MatPlotLib to graph it directly, or you can convert it back to DataFrame and have Pandas do it for you.
# 
# Since we've already demonstrated how to plot directly with MatPlotLib in `Module4/assignment1.ipynb`, this time we'll show you how to convert your transformed data back into to a Pandas Dataframe and have Pandas plot it from there.

# In[13]:

# Since we transformed via PCA, we no longer have column names; but we know we
# are in `principal-component` space, so we'll just define the coordinates accordingly:
ax = drawVectors(T, pca.components_, df.columns.values, plt, scaleFeatures)
T  = pd.DataFrame(T)

T.columns = ['component1', 'component2']
T.plot.scatter(x='component1', y='component2', marker='o', c=labels, alpha=0.75, ax=ax)

plt.show()


# In[ ]:



