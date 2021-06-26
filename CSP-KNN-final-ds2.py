import scipy.io
import scipy.signal
import numpy as np
import scipy.linalg as splg
from numpy import linalg
import scipy.io
import scipy.signal

def load_data(filename):

    m = scipy.io.loadmat(filename, struct_as_record=True)

    sample_rate = m['h']['SampleRate'][0][0][0][0]

    # load RAW EEG signal
    EEG = m['s'].T

    EEG = EEG[0:22, :]

    array_sum = np.sum(EEG)
    array_has_nan = np.isnan(array_sum)

    print(array_has_nan)

    # obtain dimensions of raw EEG data
    nchannels, nsamples = EEG.shape

    # obtain channel names
    # channel_names = m['h']['SampleRate'][0][0][0][0]

    event_onsets = m['h'][0][0][38]
    event_codes = m['h'][0][0][37]
    labels = np.zeros((1, nsamples), int)
    labels[0, event_onsets] = event_codes

    trials = {}

    cl_lab = ['left', 'right', 'foot', 'tongue']
    cl1 = cl_lab[0]
    cl2 = cl_lab[1]
    cl3 = cl_lab[2]
    cl4 = cl_lab[3]

    win = np.arange(int(2.1 * sample_rate), int(6.5 * sample_rate)) # 2.1 -- 6.5, 1.5 -- 5.9
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

            array_sum = np.sum(trials[cl][:, :, i])
            array_has_nan = np.isnan(array_sum)

            if array_has_nan == True:
                print(i)
                error = trials[cl][:, :, i]
                trials[cl][:, :, i] = trials[cl][:, :, i-1]

    return trials, cl1, cl2, sample_rate, nchannels, nsamples

trials, cl1, cl2, sample_rate, nchannels, nsamples = load_data('A01T.mat')

def bandpass(trials, lo, hi, smpl_rate):

    # filter signal between the high and low cutoff frequencies
    a, b = scipy.signal.iirfilter(6, [lo / (smpl_rate / 2.0), hi / (smpl_rate / 2.0)])

    # Applying the filter to each trial
    ntrials = trials.shape[2]
    nchannels = trials.shape[0]
    no_samples = trials.shape[1]
    trials_filt = np.zeros((nchannels, no_samples, ntrials))
    for i in range(ntrials):
        trials_filt[:, :, i] = scipy.signal.filtfilt(a, b, trials[:, :, i], axis=1)

    return trials_filt


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

# load data from matlab file
# trials, cl1, cl2, sample_rate = load_data('BCICIV_calib_ds1g.mat')

# filter the trials for the mu frequency band
trials_filt = {cl1: bandpass(trials[cl1], 8, 15, sample_rate),
               cl2: bandpass(trials[cl2], 8, 15, sample_rate)}

train_percentage = 0.5

train_index = int(np.ceil(trials[cl1].shape[2]*train_percentage))

train = {cl1: trials_filt[cl1][:, :, :train_index],
            cl2: trials_filt[cl2][:, :, :train_index]}

test = {cl1: trials_filt[cl1][:, :, train_index:],
        cl2: trials_filt[cl2][:, :, train_index:]}

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

class_prediction_right = prediction(training.T, test[cl1].T, type, 13)
class_prediction_left = prediction(training.T, test[cl2].T, type, 13)

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