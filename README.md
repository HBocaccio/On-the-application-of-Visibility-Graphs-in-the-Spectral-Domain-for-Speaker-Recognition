
# On-the-application-of-Visibility-Graphs-in-the-Spectral-Domain-for-Speaker-Recognition

This repository contains the code and data used for the speaker recognition trough graphs-based variables. The analysis applies tree-based models ensambles, specifically Random Forest, to the features obtained from visibility graphs in the spectral domain for audio recordings of Spanish vowels vocalizations.

## Data

The dataset consists of audio recordings from 7 male Spanish-speaking adult participants, with ages in the range of (37.86 ± 5.15) years. The recordings contain isolated vocalizations of the five Spanish vowels, with each vowel repeated for at least 30 seconds. The audio files were preprocessed (mono, WAV, 11025 Hz) and segmented into 890 clean audio chunks.

## Preprocessing

- Converted to mono and resampled to 11025 Hz.
- Spectral gating noise reduction applied.
- Segmentation performed using Pydub’s "split_on_silence".

## Spectral Functions

Formants were extracted using Linear Predictive Coding (LPC) with an order of 13. The frequency response was computed over a range of 0 to 5512 Hz, yielding log power spectral functions for each segment. Sensitivity analysis was performed with LPC orders ranging from 10 to 20.

### Selection of Representative Spectra  

To identify the most representative spectral profiles for each speaker and vowel, we applied a community detection approach based on spectral similarity. We computed pairwise correlations between log power spectral functions, binarized them using a threshold of 0.9, and constructed an adjacency matrix. Using the BCT toolbox, we detected communities and selected the largest one (giant component) as the most characteristic spectral patterns, retaining 83% of the total spectra. To assess robustness, we repeated this process with correlation thresholds from 0.5 to 0.95. The consistency of results across different thresholds demonstrated the method’s stability despite spectral variability.

## Visibility Graph Construction

We constructed visibility graphs from the spectral profiles of each vowel segment using the **Natural Visibility Graph (NVG)** algorithm introduced by Lacasa et al. (2008). In this framework, each log power spectral profile (i.e., log(|H(f)|²)) is treated as a one-dimensional sequence, where each node corresponds to the amplitude at a discrete frequency bin. An edge is established between two nodes (a, b), with $a < b$, if every intermediate node $c$ satisfies the **visibility criterion**:

```math
y_c < y_b + (y_a - y_b) \cdot \frac{t_b - t_c}{t_b - t_a}
```

Here, $y_i$ denotes the spectral amplitude at frequency index $t_i$, and the inequality must hold for all $t_c$ between $t_a$ and $t_b$. This ensures that node $a$ "sees" node $b$ without obstruction.

To efficiently compute these graphs across the dataset, we used a divide-and-conquer implementation based on Lan et al. (2015), adapted to operate directly on the log-magnitude spectral functions. This algorithm achieves optimal performance for general time series.

## Graph-Based Feature Extraction

After transforming each spectral profile into a visibility graph, we computed four graph-theoretical metrics to capture its structural properties:

* **Link density**: the proportion of existing edges to all possible edges, indicating overall connectivity.
* **Average shortest path length**: the mean number of steps along the shortest paths between all pairs of nodes, reflecting the efficiency of the network.
* **Clustering coefficient**: the tendency of nodes to form tightly connected neighborhoods.
* **Modularity**: the strength of division of the graph into distinct communities with dense internal links and sparse external ones.

These metrics were computed from the unweighted and undirected adjacency matrices using functions from the Brain Connectivity Toolbox (BCT) for Python. Together, they offer a compact, interpretable summary of the spectral structure of each vowel segment and serve as input features for the classification models developed in this work.

### Train and Test

The acoustic data segments for each subject and vowel were randomly divided into training (40%), validation (30%), and test (30%) sets. For this data split, we combined the metrics associated with the five vowels, randomly selecting segments of each vowel while ensuring that the combinations were made within each set to prevent data leakage. Thus, for each combination, we obtained a feature array containing the graph-based metrics from one example audio segment for each vowel, which could then be assigned to a corresponding subject label. We performed a total of 1000 combinations per subject for each data set, ensuring that this value was lower than the total number of possible combinations based on the available segments. These combined feature arrays were then used as inputs to the model throughout the different stages of the process.

## Models

To train and evaluate our speaker recognition model, we performed 10 runs with different random splits of training and test datasets. For the model, we employed an ensemble of decision trees using the Random Forest algorithm. The attribute vectors derived from the graph-theory metrics served as input features, while the target labels corresponded to the identities of the speakers. We trained the models using a supervised learning framework. Hyperparameters were optimized via grid search. Performance was evaluated using precision, recall, F1-score, and feature importance analysis with SHAP.

## Repository Structure

### Variables

- **`variables/vg_metrics/`**: This directory contains a subdirectory for each LPC order used. The corresponding spectral functions are in `H_functions.txt`, which serves as the basis for all analyses. It also includes the indices of the representative spectra and the graph-based metrics extracted from them. All these variables were obtained using a correlation threshold of 0.9 for selecting representative spectra.

- **`variables/models/`**: This directory contains the variables obtained during the training and evaluation of the models. It is structured into subdirectories, each corresponding to an LPC order used for spectral function computation. Within each of these, there are 10 subdirectories, each representing one of the 10 trained models, corresponding to different data splits. These subdirectories contain files with information on data partitioning, evaluation results of the trained models, predicted values, performance metrics, and extracted feature importances.

- **`variables/models_thresholds/order_13/`**: This directory contains the files related to the analysis performed for different correlation threshold values used in the selection of representative spectra.

### Code

- **`run_vowels.py`**: Main script that coordinates the entire speaker recognition analysis, including data loading, preprocessing, feature extraction, and model training.  

- **`vowels_vg_metrics.py`**: Computes graph-based visibility metrics from spectral profiles, transforming spectral representations into graphical structures and extracting relevant topological properties for speaker recognition.

- **`vowels_select_spectra.py`**: Selects the most representative spectra for each speaker and vowel using community detection techniques based on spectral similarity, ensuring that the most informative features are used.  

- **`vowels_select_spectra_thresholds.py`**: Extends `vowels_select_spectra.py` by evaluating different thresholds in the selection of representative spectra, studying how these affect community composition and feature quality.  

- **`vowels_features.py`**: Contains functions for extracting spectral features from vowel recordings, such as formant computation and spectral profile generation.  

- **`vowels_features_thresholds.py`**: Similar to `vowels_features.py`, but designed to assess the robustness of extracted features under different correlation thresholds, analyzing their impact on model performance.  

- **`vowels_models_rf.py`**: Implements training and evaluation of Random Forest models for speaker recognition, including hyperparameter optimization and cross-validation.  

- **`vowels_models_rf_permutation_importance.py`**: Computes feature importance using the permutation method in the Random Forest model to determine which variables have the greatest impact on predictions.  

- **`vowels_models_rf_shap.py`**: Uses SHAP (SHapley Additive exPlanations) to interpret the contribution of each feature to the Random Forest model's decisions, providing detailed explanations of predictions.  

- **`vowels_models_rf_thresholds.py`**: Evaluates the performance of the Random Forest model under different correlation thresholds in the features, analyzing model stability and generalization.  

### Results

- **`run_figures.ipynb`**: This file is a Jupyter Notebook containing code blocks used for visualizing the results and creating figures.

## Usage

To reproduce the main analysis, follow the order of code files listed below, as presented in the 'run_vowels.py' file:

- 'vowels_vg_metrics.py'
- 'vowels_select_spectra.py'
- 'vowels_features.py'
- 'vowels_models_rf.py'
- 'vowels_models_rf_shap.py'

Afterward, you can utilize the 'run_figures.ipynb' notebook to visualize the results.

## Requirements

The following Python packages are required to run the analysis on this repository:

- numpy
- os
- pandas
- bct
- sklearn
- ast
- shap

Other tools used in this work:

- scipy
- pydub
- librosa
- soundfile
- noisereduce
- matplotlib
- seaborn

## How to cite



