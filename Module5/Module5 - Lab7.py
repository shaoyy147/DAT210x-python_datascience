
# coding: utf-8

# # DAT210x - Programming with Python for DS

# ## Module5- Lab7

# In[1]:

import random, math
import pandas as pd
import numpy as np
import scipy.io

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib

matplotlib.style.use('ggplot') # Look Pretty


# Leave this alone until indicated:
Test_PCA = False


# ### A Convenience Function

# This method is for your visualization convenience only. You aren't expected to know how to put this together yourself, although you should be able to follow the code by now:

# In[2]:

def plotDecisionBoundary(model, X, y):
    print("Plotting...")

    fig = plt.figure()
    ax = fig.add_subplot(111)

    padding = 0.1
    resolution = 0.1

    #(2 for benign, 4 for malignant)
    colors = {2:'royalblue', 4:'lightsalmon'} 


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
    plt.contourf(xx, yy, Z, cmap=plt.cm.seismic)
    plt.axis('tight')

    # Plot your testing points as well...
    for label in np.unique(y):
        indices = np.where(y == label)
        plt.scatter(X[indices, 0], X[indices, 1], c=colors[label], alpha=0.8)

    p = model.get_params()
    plt.title('K = ' + str(p['n_neighbors']))
    plt.show()


# ### The Assignment

# Load in the dataset, identify nans, and set proper headers. Be sure to verify the rows line up by looking at the file in a text editor.

# In[3]:

# .. your code here ..
df = pd.read_csv(r'E:\Python_SYY\edx\DAT210x\Module5\Datasets\breast-cancer-wisconsin.data', header = None, 
                names = ['sample', 'thickness', 'size', 'shape', 'adhesion', 'epithelial', 'nuclei', 'chromatin', 'nucleoli', 'mitoses', 'status'])


# Copy out the status column into a slice, then drop it from the main dataframe. Always verify you properly executed the drop by double checking (printing out the resulting operating)! Many people forget to set the right axis here.
# 
# If you goofed up on loading the dataset and notice you have a `sample` column, this would be a good place to drop that too if you haven't already.

# In[4]:

# .. your code here ..
label = df.status
df = df.drop(axis = 1, labels = ['sample', 'status'])


# With the labels safely extracted from the dataset, replace any nan values with the mean feature / column value:

# In[5]:

df.nuclei = pd.to_numeric(df.nuclei, errors = 'coerce')


# In[6]:

# .. your code here ..
for cname in df.columns:
    df[cname] = df[cname].fillna(df[cname].mean())


# Do train_test_split. Use the same variable names as on the EdX platform in the reading material, but set the random_state=7 for reproducibility, and keep the test_size at 0.5 (50%).

# In[61]:

# .. your code here ..
from sklearn.model_selection import train_test_split
data_train, data_test, label_train, label_test = train_test_split(df, label, test_size=0.5, random_state=7)


# Experiment with the basic SKLearn preprocessing scalers. We know that the features consist of different units mixed in together, so it might be reasonable to assume feature scaling is necessary. Print out a description of the dataset, post transformation. Recall: when you do pre-processing, which portion of the dataset is your model trained upon? Also which portion(s) of your dataset actually get transformed?

# In[62]:

# .. your code here ..
from sklearn.preprocessing import RobustScaler
norm = RobustScaler()
norm.fit(data_train)
data_train = norm.transform(data_train)
data_test = norm.transform(data_test)


# ### Dimensionality Reduction

# PCA and Isomap are your new best friends

# In[63]:

model = None

if Test_PCA:
    print('Computing 2D Principle Components')
    # TODO: Implement PCA here. Save your model into the variable 'model'.
    # You should reduce down to two dimensions.
    
    # .. your code here ..
    from sklearn.decomposition import PCA
    pca = PCA(n_components = 2)
    pca.fit(data_train)
    data_train = pca.transform(data_train)
    data_test = pca.transform(data_test)

else:
    print('Computing 2D Isomap Manifold')
    # TODO: Implement Isomap here. Save your model into the variable 'model'
    # Experiment with K values from 5-10.
    # You should reduce down to two dimensions.

    # .. your code here ..
    from sklearn import manifold
    iso = manifold.Isomap(n_components = 2, n_neighbors = 5)
    iso.fit(data_train)
    data_train = iso.transform(data_train)
    data_test = iso.transform(data_test)


# Train your model against data_train, then transform both `data_train` and `data_test` using your model. You can save the results right back into the variables themselves.

# Implement and train `KNeighborsClassifier` on your projected 2D training data here. You can name your variable `knmodel`. You can use any `K` value from 1 - 15, so play around with it and see what results you can come up. Your goal is to find a good balance where you aren't too specific (low-K), nor are you too general (high-K). You should also experiment with how changing the weights parameter affects the results.

# In[65]:

# .. your code here ..
from sklearn.neighbors import KNeighborsClassifier
knmodel = KNeighborsClassifier(n_neighbors = 5)
knmodel.fit(data_train, label_train)


# Be sure to always keep the domain of the problem in mind! It's WAY more important to errantly classify a benign tumor as malignant, and have it removed, than to incorrectly leave a malignant tumor, believing it to be benign, and then having the patient progress in cancer. Since the UDF weights don't give you any class information, the only way to introduce this data into SKLearn's KNN Classifier is by "baking" it into your data. For example, randomly reducing the ratio of benign samples compared to malignant samples from the training set.

# Calculate and display the accuracy of the testing set:

# In[66]:

# .. your code changes above ..
knmodel.score(data_test, label_test)


# In[67]:

plotDecisionBoundary(knmodel, data_test, data_test)
plt.show()

