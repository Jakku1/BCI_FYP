import numpy as np
import scipy.io
import scipy.linalg as splg
import time

def load_data(file):

    m = scipy.io.loadmat(file, struct_as_record=True)

    # SciPy.io.loadmat does not deal well with Matlab structures, resulting in lots of
    # extra dimensions in the arrays. This makes the code a bit more cluttered

    print("Loading Data...")

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

    # Dictionary to store the trials in, each class gets an entry
    trials = {}

    # The time window (in samples) to extract for each trial, here 0.5 -- 2.5 seconds
    # win = np.arange(int(0.5 * sample_rate), int(2.5 * sample_rate))

    win = np.arange(int(0.5 * sample_rate), int(2.5* sample_rate))

    # Length of the time window
    nsamples = len(win)

    print("Using a window length of ", nsamples, " samples")

    # Loop over the classes (right, foot)
    for cl, code in zip(cl_lab, np.unique(event_codes)):

        # Extract the onsets for the class
        cl_onsets = event_onsets[event_codes == code]

        # Allocate memory for the trials
        trials[cl] = np.zeros((nchannels, nsamples, len(cl_onsets)))

        # Extract each trial
        for i, onset in enumerate(cl_onsets):
            trials[cl][:, :, i] = EEG[:, win + onset]

    # Some information about the dimensionality of the data (channels x time x trials)
    print('Shape of trials[cl1]:', trials[cl1].shape)
    print('Shape of trials[cl2]:', trials[cl2].shape)

    return trials, cl1, cl2

import matplotlib.pyplot as plt

import scipy.signal


def bandpass(trials, lo, hi, sample_rate):
    '''
    Designs and applies a bandpass filter to the signal.

    Parameters
    ----------
    trials : 3d-array (channels x samples x trials)
        The EEGsignal
    lo : float
        Lower frequency bound (in Hz)
    hi : float
        Upper frequency bound (in Hz)
    sample_rate : float
        Sample rate of the signal (in Hz)

    Returns
    -------
    trials_filt : 3d-array (channels x samples x trials)
        The bandpassed signal
    '''

    # The iirfilter() function takes the filter order: higher numbers mean a sharper frequency cutoff,
    # but the resulting signal might be shifted in time, lower numbers mean a soft frequency cutoff,
    # but the resulting signal less distorted in time. It also takes the lower and upper frequency bounds
    # to pass, divided by the niquist frequency, which is the sample rate divided by 2:
    a, b = scipy.signal.iirfilter(6, [lo / (sample_rate / 2.0), hi / (sample_rate / 2.0)])

    # Applying the filter to each trial
    ntrials = trials.shape[2]
    nchannels = trials.shape[0]
    nsamples = trials.shape[1]
    trials_filt = np.zeros((nchannels, nsamples, ntrials))
    for i in range(ntrials):
        trials_filt[:, :, i] = scipy.signal.filtfilt(a, b, trials[:, :, i], axis=1)

    return trials_filt

from numpy import linalg

def CSP(left, right):

    # calculate the mean covariance matrix for each class
    cov_left = covariance(left)
    cov_right = covariance(right)

    # calculate the eigenvalues and eigenvectors for the combined covariance matrices
    evals, evecs = splg.eig(cov_left+cov_right)

    #  calculate the whitening transformation
    P = evecs.dot(np.diag(evals.real**-0.5))

    #calculate the singular value decomposition
    B, _, _ = linalg.svd(P.T.dot(cov_right).dot(P))

    # calculate the projection matrix
    W = P.dot(B)

    # return the projection matrix
    return W

def covariance(trials):

    # get statistics for input
    no_channels = trials.shape[0]
    no_samples = trials.shape[1]
    no_trials = trials.shape[2]

    # store covariance matrix for each trial
    covariances = []

    # calculate covariance matrix for each trial
    for i in range(no_trials):
        product = trials[:, :, i].dot(trials[:, :, i].T)
        covariances.append(product/product.trace())

    # calculate mean of all covariance matrices along the correct axis
    cov_mean = np.mean(covariances, axis=0)
    return cov_mean

def apply_projection(W, trials, components):

    # get statistics for input
    no_channels = trials.shape[0]
    no_samples = trials.shape[1]
    no_trials = trials.shape[2]

    # create empty matrix to store transformed trials
    csp_trials = np.zeros((no_channels, no_samples, no_trials))

    # project each trial with the projection matrix
    for i in range(no_trials):
        csp_trials[:,:,i] = W.T.dot(trials[:,:,i])

    if components == 2:
        comps = np.array([0, -1])
    elif components == 4:
        comps = np.array([0, 1, -2, -1])
    elif components == 6:
        comps = np.array([0, 1, 2, -3, -2, -1])
    elif components == 8:
        comps = np.array([0, 1, 2, 3, -4, -3, -2, -1])
    else:
        print("Too many components, please choose an even number between 2 and 8")
        return -1

    csp_trials = csp_trials[comps,:,:]

    log_variance = np.log(np.var(csp_trials, axis=1))

    return log_variance

# SVM

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

def plot_scatter(left, right):
    plt.figure()
    plt.scatter(left[0,:], left[-1,:], color='b')
    plt.scatter(right[0,:], right[-1,:], color='r')
    plt.xlabel('Last component')
    plt.ylabel('First component')



# load data for all subjects
subject1 = 'BCICIV_calib_ds1a.mat'
subject2 = 'BCICIV_calib_ds1b.mat'
subject3 = 'BCICIV_calib_ds1c.mat'
subject4 = 'BCICIV_calib_ds1d.mat'
subject5 = 'BCICIV_calib_ds1e.mat'
subject6 = 'BCICIV_calib_ds1f.mat'
subject7 = 'BCICIV_calib_ds1g.mat'

# define library function keys
cl1 = 'left'
cl2 = 'right'

# trials_1 = load_data(subject1)
trials_2, cl1_2, cl2_2 = load_data(subject2)
trials_3, cl1_3, cl2_3 = load_data(subject3)
trials_4, cl1_4, cl2_4 = load_data(subject4)
# trials_5 = load_data(subject5)
trials_6, cl1_6, cl2_6 = load_data(subject6)
trials_7, cl1_7, cl2_7 = load_data(subject7)
print(cl1_2, cl2_2)
print(cl1_3, cl2_3)
print(cl1_4, cl2_4)
print(cl1_7, cl2_7)

# add all trials to the same library
trials = [trials_2, trials_3, trials_4, trials_7]
all_trials = {}
for k in trials_2.keys():
    all_trials[k] = np.concatenate(list(all_trials[k] for all_trials in trials), axis=2)

sample_rate = 100

# Apply a band pass filter to the data
trials_filt = {cl1: bandpass(all_trials[cl1], 8, 15, sample_rate),
               cl2: bandpass(all_trials[cl2], 8, 15, sample_rate)}


from sklearn.model_selection import KFold
folds = KFold(n_splits=4, random_state=None, shuffle=True)

scores = []

index_list = list(range(0, all_trials[cl1].shape[2]))

for train_index, test_index in folds.split(index_list):
    train = {cl1: trials_filt[cl1][:, :, train_index],
            cl2: trials_filt[cl2][:, :, train_index]}

    test = {cl1: trials_filt[cl1][:, :, test_index],
            cl2: trials_filt[cl2][:, :, test_index]}

    W = CSP(train[cl1], train[cl2])

    train[cl1] = apply_projection(W, train[cl1], 2)
    train[cl2] = apply_projection(W, train[cl2], 2)
    test[cl1] = apply_projection(W, test[cl1], 2)
    test[cl2] = apply_projection(W, test[cl2], 2)

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
        if x == 1:
            correct_1 += 1

    for i in class_prediction_left:
        if i == 0:
            correct_2 += 1


    print("class 1: accuracy", correct_1*100/test[cl1].shape[1] )

    print("class 2: accuracy", correct_2*100/test[cl2].shape[1])

    print("Accuracy: ", (correct_2 + correct_1)*100/(test[cl1].shape[1]+test[cl2].shape[1]))

    time.sleep(2)