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
    """
        Converts a number string to Dutch words (0-999).

        Args:
            number_string (str): Number to convert (e.g., "42")

        Returns:
            str: Dutch word representation (e.g., "tweeënveertig")
                Returns original string for numbers >= 1000
    """
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


def word_to_phonemes(word: str) -> list:
    """
    Convert a Dutch word to a list of phonemes.

    Args:
        word (str): Dutch word to convert

    Returns:
        list: List of phonemes using IPA symbols
    """
    phoneme_map = {
        'aa': 'aː', 'ee': 'eː', 'oo': 'oː', 'uu': 'yː',
        'ie': 'i', 'oe': 'u', 'eu': 'ø', 'ui': 'œy',
        'ij': 'ɛi', 'ei': 'ɛi', 'ou': 'ʌu', 'au': 'ʌu',
        'ng': 'ŋ', 'ch': 'x', 'sch': 'sx',
        'a': 'ɑ', 'e': 'ɛ', 'i': 'ɪ', 'o': 'ɔ', 'u': 'ʏ',
        'b': 'b', 'd': 'd', 'f': 'f', 'g': 'ɣ', 'h': 'h',
        'j': 'j', 'k': 'k', 'l': 'l', 'm': 'm', 'n': 'n',
        'p': 'p', 'r': 'r', 's': 's', 't': 't', 'v': 'v',
        'w': 'ʋ', 'z': 'z'
    }

    word = word.lower()
    i = 0
    result = []

    while i < len(word):
        # Check trigraphs
        if i <= len(word) - 3 and word[i:i + 3] in phoneme_map:
            result.append(phoneme_map[word[i:i + 3]])
            i += 3
        # Check digraphs
        elif i <= len(word) - 2 and word[i:i + 2] in phoneme_map:
            result.append(phoneme_map[word[i:i + 2]])
            i += 2
        # Single characters
        elif word[i] in phoneme_map:
            result.append(phoneme_map[word[i]])
            i += 1
        else:
            result.append(word[i])
            i += 1

    return result


def create_phoneme_mappings(word2phoneme):
    """
    Create phoneme-to-integer and integer-to-phoneme mappings including PAD token.

    Args:
        word2phoneme (dict): Dictionary mapping words to their phonemes

    Returns:
        tuple: (phoneme_to_int, int_to_phoneme) dictionaries
    """
    # Get unique phonemes from word_to_phoneme dictionary
    PAD_token = ' '

    # Create set of unique phonemes
    phonemes = set()
    for phoneme_list in word2phoneme.values():
        phonemes.update(phoneme_list)

    # Create phoneme to integer mapping
    phoneme_to_int = {phoneme: i + 1 for i, phoneme in enumerate(sorted(phonemes))}
    phoneme_to_int[PAD_token] = 0  # Add PAD token

    # Create integer to phoneme mapping
    int_to_phoneme = {v: k for k, v in phoneme_to_int.items()}

    return phoneme_to_int, int_to_phoneme


def convert_words_to_phoneme_matrix(words, seq_len, word2phoneme):
    """
    Convert words to phoneme integer matrix using word_to_phoneme dictionary.

    Args:
        words (list): List of words to convert
        seq_len (int): Maximum sequence length
        word2phoneme (dict): Dictionary mapping words to phonemes

    Returns:
        numpy.ndarray: Matrix of integer-encoded phonemes
    """
    PAD_token = ' '

    # Get phoneme mappings
    phoneme_to_int, _ = create_phoneme_mappings(word2phoneme)

    # Initialize matrix with PAD token
    phoneme_mat = np.full((len(words), seq_len),
                          phoneme_to_int[PAD_token], dtype=int)

    # Fill the matrix
    for i, word in enumerate(words):
        # Get phonemes for the word
        if word in word2phoneme:
            phoneme_list = word2phoneme[word]

            # Add phonemes up to seq_len
            for j, phoneme in enumerate(phoneme_list[:seq_len]):
                if phoneme in phoneme_to_int:
                    phoneme_mat[i, j] = phoneme_to_int[phoneme]

    return phoneme_mat



