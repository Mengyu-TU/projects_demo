from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from collections import Counter
import re
import string
import numpy as np


def normalize_eeg_features(X):
    """
    Normalize EEG features using StandardScaler.

    Parameters:
    -----------
    X : numpy.ndarray
        Input feature array of shape (num_unique_words, num_timesteps, num_channels)

    Returns:
    --------
    X_normalized : numpy.ndarray
        Normalized feature array of the same shape as X
    """
    # Reshape X to 2D for normalization
    X_reshaped = X.reshape(-1, X.shape[-1]) # (100, 88, 635) --> (8800, 635)

    # Initialize and fit StandardScaler
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X_reshaped)

    # Reshape back to original dimensions
    X_normalized = X_normalized.reshape(X.shape)

    return X_normalized


def apply_pca(X, n_components=100):
    """
    Apply PCA to the input data and return the first n_components.

    Parameters:
    -----------
    X : numpy.ndarray
        Input feature array of shape (num_samples, num_features)
    n_components : int, optional
        Number of principal components to retain (default is 100)

    Returns:
    --------
    X_pca : numpy.ndarray
        Transformed feature array of shape (num_samples, n_components)
    """
    # Reshape X (shape is num_samples, num_timesteps, num_channels)

    # X_reshaped = X.reshape(X.shape[0], -1) # (100, 88, 635) --> (8800, 635)
    X_reshaped = X.reshape(-1, X.shape[-1])

    # Initialize and fit PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_reshaped)
    X_pca = X_pca.reshape(X.shape[0], X.shape[1], -1)

    return X_pca


def dutch_number_to_words(number_string):
    dutch_units = ['nul', 'een', 'twee', 'drie', 'vier', 'vijf', 'zes', 'zeven', 'acht', 'negen']
    dutch_teens = ['tien', 'elf', 'twaalf', 'dertien', 'veertien', 'vijftien', 'zestien', 'zeventien', 'achttien', 'negentien']
    dutch_tens = ['', '', 'twintig', 'dertig', 'veertig', 'vijftig', 'zestig', 'zeventig', 'tachtig', 'negentig']

    num = int(number_string)
    if 0 <= num < 10:
        return dutch_units[num]
    elif 10 <= num < 20:
        return dutch_teens[num - 10]
    elif 20 <= num < 100:
        tens, units = divmod(num, 10)
        if units == 0:
            return dutch_tens[tens]
        else:
            return f"{dutch_units[units]}en{dutch_tens[tens]}"
    elif 100 <= num < 1000:
        hundreds, remainder = divmod(num, 100)
        if remainder == 0:
            return f"{dutch_units[hundreds]}honderd"
        else:
            return f"{dutch_units[hundreds]}honderd{dutch_number_to_words(str(remainder))}"
    else:
        return number_string  # Return original string for numbers 1000 and above


def clean_and_convert_to_dutch(word):
    # Remove symbols, keeping only alphanumeric characters
    cleaned_word = re.sub(r'[^a-zA-Z0-9]', '', word)

    # If the cleaned word is a number, convert it to Dutch
    if cleaned_word.isdigit():
        return dutch_number_to_words(cleaned_word)
    else:
        return cleaned_word



def process_strings(y_cleaned, seq_len=15):
    """
    Process a list of cleaned strings into a character matrix.

    Args:
    y_cleaned (list): List of cleaned strings to process.
    seq_len (int): Maximum sequence length (default: 15).

    Returns:
    tuple: (y_char_mat, char_to_int, int_to_char)
    """
    # Define special token
    PAD_token = ''

    # Create a dictionary to map characters to integers
    char_to_int = {char: i + 3 for i, char in enumerate(string.ascii_lowercase)}
    char_to_int[PAD_token] = 0

    # Create a dictionary to map integers to characters
    int_to_char = {v: k for k, v in char_to_int.items()}

    # Initialize the output matrix
    y_char_mat = np.full((len(y_cleaned), seq_len), char_to_int[PAD_token], dtype=int)

    # Fill the matrix
    for i, strings in enumerate(y_cleaned):
        # Add characters from the string
        for j, char in enumerate(strings[:seq_len]):
            y_char_mat[i, j] = char_to_int[char]

    return y_char_mat, char_to_int, int_to_char


