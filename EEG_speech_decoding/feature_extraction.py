import numpy as np
import numpy.matlib as matlib
import scipy
import scipy.signal
import scipy.stats
import scipy.io.wavfile
import scipy.fftpack


#Small helper function to speed up the hilbert transform by extending the length of data to the next power of 2
hilbert3 = lambda x: scipy.signal.hilbert(x, scipy.fftpack.next_fast_len(len(x)),axis=0)[:len(x)]


def extractHG(data, sr, windowLength=0.05, frameshift=0.01):
    """
    Window data and extract frequency-band envelope using the hilbert transform

    Parameters
    ----------
    data: array (samples, channels)
        EEG time series
    sr: int
        Sampling rate of the data
    windowLength: float
        Length of window (in seconds) in which spectrogram will be calculated
    frameshift: float
        Shift (in seconds) after which next window will be extracted
    Returns
    ----------
    feat: array (windows, channels)
        Frequency-band feature matrix
    """
    # Linear detrend to remove slow drifts or trends in the signal that might not be relevant to the analysis.
    # It subtracts a least-squares fit of a straight line from the data along the specified axis (in this case, axis 0).
    data = scipy.signal.detrend(data,axis=0)

    # Number of windows
    numWindows = int(np.floor((data.shape[0]-windowLength*sr)/(frameshift*sr)))

    #Filter High-Gamma Band: extract Hilbert envelope
    sos = scipy.signal.iirfilter(4, [70/(sr/2),170/(sr/2)],btype='bandpass',output='sos') # filter coefficients returned in Second-Order Sections (SOS) format.
    data = scipy.signal.sosfiltfilt(sos,data,axis=0) # zero-phase filtering, which prevents phase distortion

    # Attenuate second harmonic of line noise (100 Hz in Netherlands)
    sos = scipy.signal.iirfilter(4, [98/(sr/2), 102/(sr/2)], btype='bandstop', output='sos')
    data = scipy.signal.sosfiltfilt(sos, data, axis=0)

    #Create feature space
    data = np.abs(hilbert3(data)) # Hilbert transform to obtain the analytic signal, and taking its absolute value gives the envelope of the signal.
    feat = np.zeros((numWindows,data.shape[1])) # num wind x num chnnl
    for win in range(numWindows):
        start= int(np.floor((win*frameshift)*sr))
        stop = int(np.floor(start+windowLength*sr))
        feat[win,:] = np.mean(data[start:stop,:],axis=0)
    return feat


def stackFeatures(features, modelOrder=4, stepSize=5):
    """
    Add temporal context to each window by stacking neighboring feature vectors

    Parameters
    ----------
    features: array (windows, channels)
        Feature time series
    modelOrder: int
        Number of temporal context to include prior to and after current window
    stepSize: float
        Number of temporal context to skip for each next context (to compensate for frameshift)
    Returns
    ----------
    featStacked: array (windows, feat*(2*modelOrder+1))
        Stacked feature matrix
    """
    featStacked=np.zeros( (features.shape[0]-(2*modelOrder*stepSize), (2*modelOrder+1)*features.shape[1]) )
    for fNum,i in enumerate(range(modelOrder*stepSize,features.shape[0]-modelOrder*stepSize)):
        ef=features[i-modelOrder*stepSize:i+modelOrder*stepSize+1:stepSize,:]
        featStacked[fNum,:]=ef.flatten() #Add 'F' if stacked the same as matlab
    return featStacked


def downsampleLabels(labels, sr, windowLength=0.05, frameshift=0.01):
    """
    Downsamples non-numerical data by using the mode

    Parameters
    ----------
    labels: array of str
        Label time series
    sr: int
        Sampling rate of the data
    windowLength: float
        Length of window (in seconds) in which mode will be used
    frameshift: float
        Shift (in seconds) after which next window will be extracted
    Returns
    ----------
    newLabels: array of str
        Downsampled labels
    """
    numWindows = int(np.floor((labels.shape[0]-windowLength*sr)/(frameshift*sr)))
    newLabels = []
    for w in range(numWindows):
        start = int(np.floor((w*frameshift)*sr))
        stop = int(np.floor(start+windowLength*sr))
        # Use numpy.unique instead of scipy.stats.mode
        unique, counts = np.unique(labels[start:stop], return_counts=True)
        newLabels.append(unique[np.argmax(counts)])
        # newLabels[w] = unique[np.argmax(counts)].encode("ascii", errors="ignore").decode()
    return newLabels


def nameVector(elecs, modelOrder=4):
    """
    Creates list of electrode names

    Parameters
    ----------
    elecs: array of str
        Original electrode names
    modelOrder: int
        Temporal context stacked prior and after current window
        Will be added as T-modelOrder, T-(modelOrder+1), ...,  T0, ..., T+modelOrder
        to the elctrode names
    Returns
    ----------
    names: array of str
        List of electrodes including contexts, will have size elecs.shape[0]*(2*modelOrder+1)
    """
    names = matlib.repmat(elecs.astype(np.dtype(('U', 10))),1,2 * modelOrder +1).T
    for i, off in enumerate(range(-modelOrder,modelOrder+1)):
        names[i,:] = [e[0] + 'T' + str(off) for e in elecs]
    return names.flatten()  #Add 'F' if stacked the same as matlab


def remove_empty_entries(eeg, labels):
    """
    In the data, labels have the same length as eeg. But labels are empty after certain time pts.
    This function removes these empty labels and also the corresponding eeg data.
    Input:
    eeg: np.array
    labels: np.array
    Output:
    eeg: np.array
    labels: np.array
    """
    return eeg[labels != ''], labels[labels != '']