import scipy.io
import scipy.signal
import numpy as np
import scipy.io
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import scipy.signal
from sklearn.preprocessing import MinMaxScaler

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

    cl_lab = ['left', 'right', 'foot', 'tongue']
    cl1 = cl_lab[0]
    cl2 = cl_lab[1]

    nclasses = len(cl_lab)

    # Dictionary to store the trials in, each class gets an entry
    trials = {}

    # The time window (in samples) to extract for each trial, here 0.5 -- 2.5 seconds
    # win = np.arange(int(0.5 * sample_rate), int(2.5 * sample_rate))

    win = np.arange(int(2.5 * sample_rate), int(6 * sample_rate)) # 2.1 -- 6.5, 1.5 -- 5.9
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

            array_sum = np.sum(trials[cl][:, :, i])
            array_has_nan = np.isnan(array_sum)

            if array_has_nan == True:
                print(i)
                error = trials[cl][:, :, i]
                trials[cl][:, :, i] = trials[cl][:, :, i-1]


    return trials, cl1, cl2, sample_rate, nchannels, nsamples

trials, cl1, cl2, sample_rate, nchannels, nsamples = load_data('A07T.mat')

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
key_electrodes = [7, 11]
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

training = np.concatenate((train[cl1], train[cl2]), axis=2)

testing = np.concatenate((test[cl1], test[cl2]), axis=2)

labels1 = np.ones(36, dtype=int)
labels2 = np.ones(36, dtype=int)*-1
train_labels = np.concatenate((labels1, labels2))
test_labels = np.concatenate((labels1, labels2))


# Some information about the dimensionality of the data (channels x time x trials)
print('Shape of trials[cl1]:', trials[cl1].shape)
print('Shape of trials[cl2]:', trials[cl2].shape)


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
train = {cl1:train_norm[:36, :],
         cl2:train_norm[36:, :]}


test = {cl1:test_norm[:36, :],
         cl2:test_norm[36:, :]}

training = np.concatenate((train[cl1], train[cl2]), axis=1)

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
        predictions.append(np.dot(trial, weights_vec) - bias)

    return np.sign(predictions)

weights_vec, bias = SVM(0.001, 0.01, 1000, training, train_labels)

pred_1 = classify_SVM(test[cl1].T, weights_vec, bias)

pred_2 = classify_SVM(test[cl2].T, weights_vec, bias)

correct_1 = 0
correct_2 = 0

#calculate the number of correct classifications
for x in pred_1:
    if x == 1:
        correct_1 += 1

for i in pred_2:
    if i == -1:
        correct_2 += 1

# print accuracy scores
print("class 1", pred_1, "accuracy", correct_1*100/36)

print("class 2 ", pred_2, "accuracy", correct_2*100/36)

print("accuracy:", (correct_1+correct_2)*100/72)




