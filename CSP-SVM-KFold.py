import numpy as np
import scipy.io
import scipy.linalg as splg

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

def train_SVM(learning_rate, lambda_val, iters, trials, weights_vec, bias, labels):

    # run optimisation for the number of iterations
    for i in range(iters):
        # loop through each trial in the matrix
        for idx, trial in enumerate(trials.T):
            # calculate the
            value = labels[idx] * (np.dot(trial, weights_vec))
            if value >= 1:
                weights_vec -= learning_rate * (2 * lambda_val * weights_vec)
            else:
                weights_vec -= learning_rate * (2 * lambda_val * weights_vec - np.dot(trial, labels[idx]))
                bias -= learning_rate * labels[idx]

    return weights_vec, bias


def SVM(learning_rate, lambda_val, iters, trials, labels):

    samples, features = trials.T.shape

    weights_vec = np.zeros(features)
    bias = 0

    # class_labels = new_labels(labels)
    class_labels = labels

    new_weights_vec, new_bias = train_SVM(learning_rate, lambda_val, iters, trials, weights_vec, bias, class_labels)

    return new_weights_vec, new_bias


def classify_SVM(trials, weights_vec, bias):

    predictions = []

    for trial in trials:
        predictions.append(bias - np.dot(trial, weights_vec))

    return np.sign(predictions)

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

print('Shape of all_trials[cl1]:', all_trials['left'].shape)
print('Shape of all_trials[cl2]:', all_trials['right'].shape)


# split into test and train
# split percentage? 50/50, 67/33
# would be 200, 200 or 268/132

# k cross fold validation

from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

# folds = StratifiedKFold(n_splits=10)
folds = KFold(n_splits=4, random_state=None, shuffle=True)

scores = []

index_list = list(range(0, all_trials[cl1].shape[2]))

for train_index, test_index in folds.split(index_list):
    train = {cl1: trials_filt[cl1][:, :, train_index],
            cl2: trials_filt[cl2][:, :, train_index]}

    test = {cl1: trials_filt[cl1][:, :, test_index],
            cl2: trials_filt[cl2][:, :, test_index]}

    print(train[cl1].shape, train[cl2].shape, test[cl1].shape, test[cl2].shape)
    print(train[cl1].shape[2] + train[cl2].shape[2] + test[cl1].shape[2] + test[cl2].shape[2])

    W = CSP(train[cl1], train[cl2])

    train[cl1] = apply_projection(W, train[cl1], 2)
    train[cl2] = apply_projection(W, train[cl2], 2)
    test[cl1] = apply_projection(W, test[cl1], 2)
    test[cl2] = apply_projection(W, test[cl2], 2)

    labels1 = np.ones(train[cl1].shape[1], dtype=int)
    labels2 = np.ones(train[cl2].shape[1], dtype=int) * -1
    labels = np.concatenate((labels1, labels2))
    training = np.concatenate((train[cl1], train[cl2]), axis=1)

    test_labels1 = np.ones(test[cl1].shape[1], dtype=int)
    test_labels2 = np.ones(test[cl1].shape[1], dtype=int) * -1
    type = np.concatenate((test_labels2, test_labels1), axis=0)

    weights_vec, bias = SVM(0.00001, 0.01, 1000, training, labels)

    pred_1 = classify_SVM(test[cl1].T, weights_vec, bias)

    pred_2 = classify_SVM(test[cl2].T, weights_vec, bias)

    correct_1 = 0
    correct_2 = 0

    for x in pred_1:
        if x == 1:
            correct_1 += 1

    for i in pred_2:
        if i == -1:
            correct_2 += 1

    # 7 incorrect/100 approx 93% so pretty good

    print("class 1", pred_1, "accuracy", correct_1 * 100 / 50)

    print("class 2 ", pred_2, "accuracy", correct_2 * 100 / 50)

    left = train[cl1]
    right = train[cl2]

    # Calculate decision boundary (x,y)
    x = np.arange(7.5, 15, 0.1)
    y = (bias - weights_vec[0] * x) / weights_vec[1]

    plt.figure()
    plt.scatter(left[0, :], left[-1, :], color='b')
    plt.scatter(right[0, :], right[-1, :], color='r')
    plt.xlabel('Last component')
    plt.ylabel('First component')
    plt.title('Training data')

    # Plot the decision boundary
    plt.plot(x, y, linestyle='--', linewidth=2, color='k')

    plt.show()

