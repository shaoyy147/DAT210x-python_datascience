
# coding: utf-8

# # DAT210x - Programming with Python for DS

# ## Module5- Lab5

# In[1]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.style.use('ggplot') # Look Pretty


# ### A Convenience Function

# In[2]:

def plotDecisionBoundary(model, X, y):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    padding = 0.6
    resolution = 0.0025
    colors = ['royalblue','forestgreen','ghostwhite']

    # Calculate the boundaris
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_min -= x_range * padding
    y_min -= y_range * padding
    x_max += x_range * padding
    y_max += y_range * padding

    # Create a 2D Grid Matrix. The values stored in the matrix
    # are the predictions of the class at at said location
    xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                       np.arange(y_min, y_max, resolution))

    # What class does the classifier say?
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the contour map
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.terrain)

    # Plot the test original points as well...
    for label in range(len(np.unique(y))):
        indices = np.where(y == label)
        plt.scatter(X[indices, 0], X[indices, 1], c=colors[label], label=str(label), alpha=0.8)

    p = model.get_params()
    plt.axis('tight')
    plt.title('K = ' + str(p['n_neighbors']))


# ### The Assignment

# Load up the dataset into a variable called `X`. Check `.head` and `dtypes` to make sure you're loading your data properly--don't fail on the 1st step!

# In[3]:

# .. your code here ..
X = pd.read_csv(r'E:\Python_SYY\edx\DAT210x\Module5\Datasets\wheat.data')


# Copy the `wheat_type` series slice out of `X`, and into a series called `y`. Then drop the original `wheat_type` column from the `X`:

# In[13]:

# .. your code here ..
y = X.wheat_type.copy()
X = X.drop(labels = ['wheat_type'], axis = 1)


# Do a quick, "ordinal" conversion of `y`. In actuality our classification isn't ordinal, but just as an experiment...

# In[16]:

# .. your code here ..
ordered_cat= ['kama', 'canadian', 'rosa']
y = y.astype("category",ordered=True,categories=ordered_cat).cat.codes


# Do some basic nan munging. Fill each row's nans with the mean of the feature:

# In[21]:

# .. your code here ..
for cname in X.columns:
    X[cname] = X[cname].fillna(X[cname].mean())


# Split `X` into training and testing data sets using `train_test_split()`. Use `0.33` test size, and use `random_state=1`. This is important so that your answers are verifiable. In the real world, you wouldn't specify a random_state:

# In[22]:

# .. your code here ..
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 1)


# Create an instance of SKLearn's Normalizer class and then train it using its .fit() method against your _training_ data. The reason you only fit against your training data is because in a real-world situation, you'll only have your training data to train with! In this lab setting, you have both train+test data; but in the wild, you'll only have your training data, and then unlabeled data you want to apply your models to.

# In[23]:

# .. your code here ..
from sklearn.preprocessing import Normalizer
norm = Normalizer()
norm.fit(X_train)


# With your trained pre-processor, transform both your training AND testing data. Any testing data has to be transformed with your preprocessor that has ben fit against your training data, so that it exist in the same feature-space as the original data used to train your models.

# In[24]:

# .. your code here ..
X_train = norm.transform(X_train)
X_test = norm.transform(X_test)


# Just like your preprocessing transformation, create a PCA transformation as well. Fit it against your training data, and then project your training and testing features into PCA space using the PCA model's `.transform()` method. This has to be done because the only way to visualize the decision boundary in 2D would be if your KNN algo ran in 2D as well:

# In[25]:

# .. your code here ..
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
pca.fit(X_test)


# In[26]:

X_train = pca.transform(X_train)
X_test = pca.transform(X_test)


# Create and train a KNeighborsClassifier. Start with `K=9` neighbors. Be sure train your classifier against the pre-processed, PCA- transformed training data above! You do not, of course, need to transform your labels.

# In[67]:

# .. your code here ..
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(X_train, y_train)


# In[68]:

# I hope your KNeighbors classifier model from earlier was named 'knn'
# If not, adjust the following line:
plotDecisionBoundary(knn, X_train, y_train)


# Display the accuracy score of your test data/labels, computed by your KNeighbors model. You do NOT have to run `.predict` before calling `.score`, since `.score` will take care of running your predictions for you automatically.

# In[69]:

# .. your code here ..
from sklearn.metrics import accuracy_score
accuracy_score(y_test, knn.predict(X_test))
# the other way
knn.score(X_test,y_test)


# ### Bonus

# Instead of the ordinal conversion, try and get this assignment working with a proper Pandas get_dummies for feature encoding. You might have to update some of the `plotDecisionBoundary()` code.

# In[70]:

plt.show()


# In[ ]:



