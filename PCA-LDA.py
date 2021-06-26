import numpy as np
import scipy.io
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import scipy.signal
from sklearn.preprocessing import MinMaxScaler
# k Nearest Neighbour
from sklearn.neighbors import KNeighborsClassifier
from operator import itemgetter
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

import scipy.linalg as splg


m = scipy.io.loadmat('BCICIV_calib_ds1g.mat', struct_as_record=True)

# SciPy.io.loadmat does not deal well with Matlab structures, resulting in lots of
# extra dimensions in the arrays. This makes the code a bit more cluttered

sample_rate = m['nfo']['fs'][0][0][0][0]
EEG = m['cnt'].T
nchannels, nsamples = EEG.shape

channel_names = [s[0] for s in m['nfo']['clab'][0][0][0]]
event_onsets = m['mrk'][0][0][0]
event_codes = m['mrk'][0][0][1]
labels = np.zeros((1, nsamples), int)
labels[0, event_onsets] = event_codes

cl_lab = [s[0] for s in m['nfo']['classes'][0][0][0]]
cl1 = cl_lab[0]
cl2 = cl_lab[1]
nclasses = len(cl_lab)
nevents = len(event_onsets)

# Print some information
print('Shape of EEG:', EEG.shape)
print('Sample rate:', sample_rate)
print('Number of channels:', nchannels)
print('Channel names:', channel_names)
print('Number of events:', len(event_onsets))
print('Event codes:', np.unique(event_codes))
print('Class labels:', cl_lab)
print('Number of classes:', nclasses)

# print('Key Electrodes:', *itemgetter(26,28,30,43,45,49,52,53)(channel_names))
print('Key Electrodes:', *itemgetter(26, 30)(channel_names))

# Dictionary to store the trials in, each class gets an entry
trials = {}

# The time window (in samples) to extract for each trial, here 0.5 -- 2.5 seconds
win = np.arange(int(1 * sample_rate), int(3.5 * sample_rate))

# Length of the time window
nsamples = len(win)

# Loop over the classes (right, foot)
for cl, code in zip(cl_lab, np.unique(event_codes)):

    # Extract the onsets for the class
    cl_onsets = event_onsets[event_codes == code]

    # Allocate memory for the trials
    trials[cl] = np.zeros((nchannels, nsamples, len(cl_onsets)))

    # Extract each trial
    for i, onset in enumerate(cl_onsets):
        trials[cl][:, :, i] = EEG[:, win + onset]


# comment


def bandpass(trials, lo, hi, sample_rate):

    a, b = scipy.signal.iirfilter(6, [lo / (sample_rate / 2.0), hi / (sample_rate / 2.0)])

    # Applying the filter to each trial
    ntrials = trials.shape[2]
    trials_filt = np.zeros((nchannels, nsamples, ntrials))
    for i in range(ntrials):
        trials_filt[:, :, i] = scipy.signal.filtfilt(a, b, trials[:, :, i], axis=1)

    return trials_filt


        # Apply the function
trials_filt = {cl1: bandpass(trials[cl1], 8, 15, sample_rate),
                cl2: bandpass(trials[cl2], 8, 15, sample_rate)}

# Percentage of trials to use for training (50-50 split here)
train_percentage = 0.5



# Calculate the number of trials for each class the above percentage boils down to
ntrain_r = int(trials_filt[cl1].shape[2] * train_percentage)
ntrain_f = int(trials_filt[cl2].shape[2] * train_percentage)
ntest_r = trials_filt[cl1].shape[2] - ntrain_r
ntest_f = trials_filt[cl2].shape[2] - ntrain_f

# key_electrodes = [26,28,30,43,45,49,52,53]
key_electrodes = [26, 30]
# key_electrodes = []

# Splitting the frequency filtered signal into a train and test set
train = {cl1: trials_filt[cl1][:,:,:ntrain_r],
         cl2: trials_filt[cl2][:,:,:ntrain_f]}

train_reduce = {cl1: train[cl1][key_electrodes,:,:],
            cl2: train[cl2][key_electrodes,:,:]}

test = {cl1: trials_filt[cl1][:,:,ntrain_r:],
        cl2: trials_filt[cl2][:,:,ntrain_f:]}

test_reduce = {cl1: test[cl1][key_electrodes,:,:],
            cl2: test[cl2][key_electrodes,:,:]}

training = np.concatenate((train_reduce[cl1], train_reduce[cl2]), axis=2)

testing = np.concatenate((test_reduce[cl1], test_reduce[cl2]), axis=2)

labels1 = np.ones(50, dtype=int)
labels2 = np.zeros(50, dtype=int)
train_labels = np.concatenate((labels1, labels2))
test_labels = np.concatenate((labels1, labels2))


# Some information about the dimensionality of the data (channels x time x trials)
print('Shape of trials[cl1]:', trials[cl1].shape)
print('Shape of trials[cl2]:', trials[cl2].shape)


data_2d = np.array([features_2d.flatten() for features_2d in np.transpose(training, (2, 0, 1))])

print(data_2d.shape)

test_2d = np.array([features_2d.flatten() for features_2d in np.transpose(testing, (2, 0, 1))])

plt.show()

pca = PCA(n_components=15)

# pca_values = pca.fit(data_2d)
pca.fit(data_2d)

var = pca.explained_variance_ratio_

pca.components_[0]

print(np.sum(var))

img_transformed = pca.transform(data_2d)

print(img_transformed.shape)

train_ext = pca.fit_transform(data_2d)

test_ext = pca.fit_transform(test_2d)

min_max_scaler = MinMaxScaler()
train_norm = min_max_scaler.fit_transform(train_ext)
test_norm = min_max_scaler.fit_transform(test_ext)

train = {cl1:train_norm[:50, :],
         cl2:train_norm[50:, :]}

test = {cl1:test_norm[:50, :],
         cl2:test_norm[50:, :]}


def LDA(class1, class2):
    no_classes = 2

    # calculate priors for each class
    prior1 = class1.shape[0] / (class1.shape[0] + class2.shape[0])
    prior2 = class2.shape[0] / (class1.shape[0] + class2.shape[0])

    # centre each class
    class1_centre = class1 - np.mean(class1, axis=0)
    class2_centre = class2 - np.mean(class2, axis=0)

    # calcaulte the in class covariance matrix for each class
    covaraiance1 = class1_centre.T.dot(class1_centre) / (class1.shape[0] - no_classes)
    covaraiance2 = class2_centre.T.dot(class2_centre) / (class2.shape[0] - no_classes)

    prior_cov1 = prior1 * covaraiance1
    prior_cov2 = prior2 * covaraiance2

    prior_mean1 = prior1 * np.mean(class1, axis=0)
    prior_mean2 = prior1 * np.mean(class2, axis=0)

    # calculate the new weights and bias
    weights = ((np.mean(class2, axis=0) - np.mean(class1, axis=0)).dot(np.linalg.pinv(prior_cov1 + prior_cov2)))
    bias = (prior_mean1 + prior_mean2).dot(weights)

    return weights, bias

def LDA_prediction(test, weights, bias):

    # create an empty array to assign predictions
    predictions = []

    # loop through each trial
    for trial in test.T:
        # multiply the weights vector by the test vector and subtract the bias
        result = weights.dot(trial) - bias
        if result <=0:
            predictions.append(1)
        else:
            predictions.append(0)

    return np.array(predictions)


weights, bias = LDA(train[cl1], train[cl2])

confusion_matrix = np.array([
    [(LDA_prediction(test[cl1].T, weights, bias) == 1).sum(), (LDA_prediction(test[cl2].T, weights, bias) == 1).sum()],
    [(LDA_prediction(test[cl1].T, weights, bias) == 0).sum(), (LDA_prediction(test[cl2].T, weights, bias) == 0).sum()],
])

print('Confusion matrix:\n', confusion_matrix)

print('Accuracy: %.6f' % (np.sum(np.diag(confusion_matrix)) / float(np.sum(confusion_matrix))))


