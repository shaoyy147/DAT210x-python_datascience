
# coding: utf-8

# # DAT210x - Programming with Python for DS

# ## Module5- Lab10

# In[1]:

import numpy as np
import pandas as pd

from sklearn.utils.validation import check_random_state
import scipy.io.wavfile as wavfile


# Good Luck! Heh.

# ### About Audio

# Samples are Observations. Each audio file will is a single sample in our dataset.
# 
# Find more information about [Audio Samples here](https://en.wikipedia.org/wiki/Sampling_(signal_processing)).
# 
# Each .wav file is actually just a bunch of numeric samples, "sampled" from the analog signal. Sampling is a type of discretization. When we mention 'samples', we mean observations. When we mention 'audio samples', we mean the actually "features" of the audio file.
# 
# The goal of this lab is to use multi-target, linear regression to generate by extrapolation, the missing portion of the test audio file.
# 
# Each one audio_sample features will be the output of an equation, which is a function of the provided portion of the audio_samples:
# 
#     missing_samples = f(provided_samples)
# 
# You can experiment with how much of the audio you want to chop off and have the computer generate using the Provided_Portion parameter.

# Play with this. This is how much of the audio file will be provided, in percent. The remaining percent of the file will be generated via linear extrapolation.

# In[2]:

Provided_Portion = 0.25


# ### The Assignment

# You have to download the dataset (audio files) from the website: https://github.com/Jakobovski/free-spoken-digit-dataset

# Start by creating a regular Python List called `zero`:

# In[3]:

# .. your code here ..
zero = []


# Loop through the dataset and load up all 50 of the `0_jackson*.wav` files using the `wavfile.read()` method: https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.io.wavfile.read.html Be careful! `.read()` returns a tuple and you're only interested in the audio data, and not sample_rate at this point. Inside your for loop, simply append the loaded audio data into your Python list `zero`:

# In[4]:

# .. your code here ..
from os import listdir
files = listdir(r'E:\Machine Learning\Audio\RecordingNeed')
for fname in files:
    sample_rate,data = wavfile.read('E:\\Machine Learning\\Audio\\RecordingNeed\\' + fname)
    zero.append(data)


# Just for a second, convert zero into a DataFrame. When you do so, set the `dtype` to `np.int16`, since the input audio files are 16 bits per sample. If you don't know how to do this, read up on the docs here: http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html
# 
# Since these audio clips are unfortunately not length-normalized, we're going to have to just hard chop them to all be the same length. Since Pandas would have inserted NANs at any spot to make zero a  perfectly rectangular [n_observed_samples, n_audio_samples] array, do a `dropna` on the Y axis here. Then, convert one back into an NDArray using `yourarrayname.values`:

# In[5]:

# .. your code here ..
df = pd.DataFrame(zero, dtype = np.int16)
df = df.dropna(axis = 1)
zero = df.values


# It's important to know how (many audio_samples samples) long the data is now.
# 
# `zero` is currently shaped like `[n_samples, n_audio_samples]`, so get the `n_audio_samples` count and store it in a variable called `n_audio_samples`:

# In[6]:

# .. your code here ..
n_audio_samples = len(zero[0])


# Create your linear regression model here and store it in a variable called `model`. Don't actually train or do anything else with it yet:

# In[7]:

# .. your code here ..
from sklearn import linear_model
model = linear_model.LinearRegression()


# There are 50 takes of each clip. You want to pull out just one of them, randomly, and that one will NOT be used in the training of your model. In other words, the one file we'll be testing / scoring on will be an unseen sample, independent to the rest of your training set:

# In[8]:

# Leave this line alone until you've submitted your lab:
rng = check_random_state(7)

random_idx = rng.randint(zero.shape[0])
test  = zero[random_idx]
train = np.delete(zero, [random_idx], axis=0)


# Print out the shape of `train`, and the shape of `test`.
# 
# `train` will be shaped: `[n_samples, n_audio_samples]`, where `n_audio_samples` are the 'features' of the audio file 
# 
# `test` will be shaped `[n_audio_features]`, since it is a single sample (audio file, e.g. observation).

# In[9]:

# .. your code here ..
train.shape
test.shape


# The test data will have two parts, `X_test` and `y_test`.
# 
# `X_test` is going to be the first portion of the test audio file, which we will be providing the computer as input. 
# 
# `y_test`, the "label" if you will, is going to be the remaining portion of the audio file. Like such, the computer will use linear regression to derive the missing portion of the sound file based off of the training data its received!
# 
# Let's save the original `test` clip, the one you're about to delete half of, to the current directory so that you can compare it to the 'patched' clip once you've generated it. You should have already got the `sample_rate` when you were loading up the .wav files:

# In[10]:

wavfile.write('Original Test Clip.wav', sample_rate, test)


# Prepare the TEST data by creating a slice called `X_test`. It should have `Provided_Portion` * `n_audio_samples` audio sample features, taken from your test audio file, currently stored in variable `test`. In other words, grab the FIRST `Provided_Portion` * `n_audio_samples` audio features from `test` and store it in `X_test`. This should be accomplished using indexing:

# In[11]:

# .. your code here ..
X_test = test[0:int(Provided_Portion*n_audio_samples)]


# If the first `Provided_Portion` * `n_audio_samples` features were stored in `X_test`, then we need to also grab the _remaining_ audio features and store them in `y_test`. With the remaining features stored in there, we will be able to R^2 "score" how well our algorithm did in completing the sound file.

# In[12]:

# .. your code here ..
y_test = test[int(Provided_Portion*n_audio_samples)+1 : n_audio_samples-1]


# Duplicate the same process for `X_train`, `y_train`. The only differences being:
# 
# 1. Your will be getting your audio data from `train` instead of from `test`
# 2. Remember the shape of `train` that you printed out earlier? You want to do this slicing but for ALL samples (observations). For each observation, you want to slice the first `Provided_Portion` * `n_audio_samples` audio features into `X_train`, and the remaining go into `y_test`. All of this should be doable using regular indexing in two lines of code:

# In[13]:

# .. your code here ..
X_train = train[:,0:int(Provided_Portion*n_audio_samples)]
y_train = train[:,int(Provided_Portion*n_audio_samples)+1 : n_audio_samples-1]


# In[14]:

X_train.shape


# SciKit-Learn gets 'angry' if you don't supply your training data in the form of a 2D dataframe shaped like `[n_samples, n_features]`.
# 
# So if you only have one SAMPLE, such as is our case with `X_test`, and `y_test`, then by calling `.reshape(1, -1)`, you can turn `[n_features]` into `[1, n_features]` in order to appease SciKit-Learn.
# 
# On the other hand, if you only have one FEATURE, you can alternatively call `.reshape(-1, 1)` on your data to turn `[n_samples]` into `[n_samples, 1]`.
# 
# Reshape X_test and y_test as [1, n_features]:

# In[21]:

# .. your code here ..
X_test = X_test.reshape(1,-1)
y_test = y_test.reshape(1,-1)


# In[22]:

X_test.shape


# Fit your model using your training data and label:

# In[17]:

# .. your code here ..
model.fit(X_train, y_train)


# Use your model to predict the `label` of `X_test`. Store the resulting prediction in a variable called `y_test_prediction`:

# In[23]:

# .. your code here ..
y_test_prediction = model.predict(X_test)


# SciKit-Learn will use float64 to generate your predictions so let's take those values back to int16, which is what our .wav files expect:

# In[24]:

y_test_prediction = y_test_prediction.astype(dtype=np.int16)


# In[29]:

y_test_prediction


# In[30]:

y_test


# Score how well your prediction would do for some good laughs, by passing in your test data and test label `y_test`:

# In[28]:

# .. your code here ..
model.score(X_test, y_test)


# In[26]:

print("Extrapolation R^2 Score: ", score)


# Let's take the first `Provided_Portion` portion of the test clip, the part you fed into your linear regression model. Then, stitch that together with the 'abomination' the predictor model generated for you and then save the completed audio clip:

# In[27]:

completed_clip = np.hstack((X_test, y_test_prediction))
wavfile.write('Extrapolated Clip.wav', sample_rate, completed_clip[0])


# Congrats on making it to the end of this crazy lab and module =) !

# In[ ]:



