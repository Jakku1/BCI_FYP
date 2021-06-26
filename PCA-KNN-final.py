import numpy as np
import scipy.io
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import scipy.signal
from sklearn.preprocessing import MinMaxScaler
from operator import itemgetter


m = scipy.io.loadmat('BCICIV_calib_ds1g.mat', struct_as_record=True)

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

train_reduce = {cl1: trials_filt[cl1][key_electrodes,:,:],
            cl2: trials_filt[cl2][key_electrodes,:,:]}

test = {cl1: trials_filt[cl1][:,:,ntrain_r:],
        cl2: trials_filt[cl2][:,:,ntrain_f:]}

test_reduce = {cl1: test[cl1][28,:,:],
            cl2: test[cl2][28,:,:]}

training = np.concatenate((train[cl1], train[cl2]), axis=2)

testing = np.concatenate((test[cl1], test[cl2]), axis=2)

labels1 = np.ones(50, dtype=int)
labels2 = np.ones(50, dtype=int)*-1
train_labels = np.concatenate((labels1, labels2))
test_labels = np.concatenate((labels1, labels2))

def euclidean_distance(x1, x2):
    """
    Calculate the euclidean distance between the test point and every other point
    :param x1: vector containing each dimension of the test variable
    :param x2: vector containing each dimension of the target variable
    :return: euclidean distance between the two points
    """
    dim_diff = []

    for i, x in enumerate(x1):
        dim_diff.append(abs(x1[i]-x2[i])**2)

    return np.sqrt(sum(dim_diff))

def manhattan_distance(x1, x2):

    dim_diff = []

    for i, x in enumerate(x1):
        dim_diff.append(abs(x1[i] - x2[i]))

    return sum(dim_diff)

def find_nearest_neighbours(train, type, test, k):
    distance_list = []
    # calculate the euclidean distance for every point
    for i, x in enumerate(train):
        distance = euclidean_distance(test, train[i])
        distance_list.append([i, distance])
    # combine class of each point with distance
    for i, x in enumerate(distance_list):
        distance_list[i].append(type[i])
        print(i)
    # sort by the distance between the points
    sorted_list = sorted(distance_list, key=lambda a_entry: a_entry[1])
    print(sorted_list)
    # select k points with the closes distances
    nearest_neighbours = [sorted_list[0:k]]

    return nearest_neighbours

def prediction(train, test, type, neighbours):

    classifications = []

    for test_point in test:
        # find the nearest neighbours to the test point
        nearest_neighbours = find_nearest_neighbours(train, type, test_point, neighbours)

        # display the nearest neighbours
        print("nearest neighbours\n", nearest_neighbours)

        sorted_type = []

        # extract the class for each of the nearest neighbours
        sorted_type = [x[2] for x in nearest_neighbours[0]]
        print("sorted type:\n", sorted_type)

        # calculate the mode of the classes
        # most votes determined the chosen class
        classifications.append(max(set(sorted_type), key=sorted_type.count))

    return classifications


data_2d = np.array([features_2d.flatten() for features_2d in np.transpose(training, (2, 0, 1))])

print(data_2d.shape)

test_2d = np.array([features_2d.flatten() for features_2d in np.transpose(testing, (2, 0, 1))])

plt.show()

pca = PCA(n_components=30)

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

labels1 = np.ones(train[cl1].shape[1], dtype=int)
labels2 = np.zeros(train[cl2].shape[1], dtype=int)
labels = np.concatenate((labels1, labels2))
training = np.concatenate((train[cl1], train[cl2]), axis=1)

test_labels1 = np.ones(test[cl1].shape[1], dtype=int)
test_labels2 = np.zeros(test[cl1].shape[1], dtype=int)
type = np.concatenate((test_labels2, test_labels1), axis=0)

class_prediction_right = prediction(training.T, test[cl1].T, labels, 13)
class_prediction_left = prediction(training.T, test[cl2].T, labels, 13)

print("right:\n", class_prediction_right)
print("left:\n", class_prediction_left)

correct_1 = 0
correct_2 = 0

for x in class_prediction_right:
    if x == 0:
        correct_1 += 1

for i in class_prediction_left:
    if i == 1:
        correct_2 += 1

# 7 incorrect/100 approx 93% so pretty good

print("class 1: accuracy", correct_1*100/50)

print("class 2: accuracy", correct_2*100/50)

print("Accuracy: ", (correct_2 + correct_1))