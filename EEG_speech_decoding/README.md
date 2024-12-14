# Phoneme-Level Speech Decoding from EEG with Sequence Models 
This project investigates the decoding of phoneme-level speech from EEG 
signals during speech production using sequence-to-sequence models. 
While traditional approaches have focused on word-level decoding, 
my work aims to achieve finer granularity by targeting phoneme-level 
reconstruction. Phonemes are the smallest units of sound in speech that 
distinguish one word from another. For example, the words "bat" and "pat" 
differ only in their initial phonemes (/b/ and /p/). 
The models can decode previously unseen words by leveraging phoneme-level 
patterns.

## Overview
This documentation covers the setup and usage of the EEG signal decoding project, which utilizes sequence-to-sequence models for signal processing and analysis.

## Environment Setup

### Prerequisites
```
Python 3.8+
PyTorch
NumPy
Pandas
scikit-learn
Jupyter Notebook
NWB (Neuroscience without border) for Python: for reading the data file from the source paper
```

### Installation
```bash
# Install dependencies
pip install torch numpy pandas scikit-learn jupyter pynwb
```

## Project Structure

### Core Files
- `custom_dataset.py` - EEG dataset handler
- `feature_extraction.py` - Feature extraction utilities
- `seq2seq_model.py` - Model implementation
- `utils_func.py` - Helper functions
- `EEG_decoding_notebook_Mengyu_TU.ipynb` - Main experiment notebook with **training logs**

## Usage Instructions

### Data 
1. Data is from Verwoert, M., Ottenhoff, M. C., Goulis, S., Colon, A. J., Wagner, L., Tousseyn, S., van
Dijk, J. P., Kubben, P. L., & Herff, C. (2022). Dataset of Speech Production in intracranial
Electroencephalography. Scientific Data, 9(1), 434. 
Available at [link](https://doi.org/10.1038/s41597-022-01542-9)


### Running Experiments
1. Launch Jupyter Notebook:
   ```bash
   jupyter notebook EEG_decoding_notebook_Mengyu_TU.ipynb
   ```
2. Follow the notebook instructions sequentially
