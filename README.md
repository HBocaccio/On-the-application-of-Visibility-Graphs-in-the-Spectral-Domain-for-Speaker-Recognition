
# On-the-application-of-Visibility-Graphs-in-the-Spectral-Domain-for-Speaker-Recognition

This repository contains the code and data used for the speaker recognition trough graphs-based variables. The analysis applies tree-based models ensambles, specifically Random Forest, to the features obtained from visibility graphs in the spectral domain for audio recordings of Spanish vowels vocalizations.

## Data

The audio data used in this study was collected from 7 male adult subjects, with ages in the range of (37.86 ± 5.15) years. Participants were instructed to use the microphone on their phones to record audio samples articulating the five Spanish vowels in vowel-separated audio recordings. In each recording session, they were directed to produce isolated vocalizations of each specific Spanish vowel, repeating it multiple times for at least half a minute. As a result, each audio file contains various samples of vocalizations of a single unique vowel, and the complete set of recordings constitutes a comprehensive collection of vocalizations representing the five vowels that comprise the Spanish language. All subjects are native Spanish speakers. The recordings were requested to be performed in a controlled environment to minimize background noise and ensure recording consistency.

### Data Pre-processing

The audio signals were first converted to mono in WAV format and resampled to a uniform sampling rate of 11025 Hz to ensure consistent data processing across all recordings. This sampling rate was chosen for its computational efficiency while preserving the relevant frequency content necessary for formant analysis. The recordings were then preprocessed to eliminate artifacts and residual background noise using spectral gating noise reduction techniques [^1]. The preprocessed audio files were segmented using the "split_on_silence" function from the Pydub Python package, which divides audio into smaller chunks by detecting pauses or silences based on a specified threshold for duration and volume. After careful review, artifactual segments were manually discarded, ensuring that each segment contained only a single vocalization of one of the five vowels. This allowed us to isolate each vowel sound for subsequent spectral function computation, resulting in a total of 890 stratified audio segments.

### Spectral functions

For the extraction of the spectral profile, we compute the speech formants by employing a linear predictive coding (LPC) approach, a well-established method in speech processing for modeling the vocal tract filter related spectral content of the speech signals [Atal & Hanauer, 1971]. The LPC coefficients were computed using the Librosa package of Python [McFee et al., 2015], with the LPC order set to 13. This selection follows the methodology of previous works [Trevisan et al., 2005], where 13 poles were sufficient to capture the main features of the formant’s envelope, offering a balance between model complexity and the accuracy of the spectral representation of the vowels. Once the LPC coefficients were obtained, we computed the frequency response of the system using the “scipy.signal.freqz” function [Virtanen et al., 2020], specifying the use of 512 discrete frequency bins to ensure a high-resolution spectral profile. Hence, the power spectral function was estimated according equation (1), using d_0=1 and d_k as the set of LPC coefficients.

\begin{equation}
H(f) = \frac{d_{0}}{1-\sum_{k=1}^{m} d_{k}e^{i k 2 \pi f \Delta}}
\label{eq1}
\end{equation}

Due to the sampling rate after resampling, the spectral profiles were computed over a frequency range of 0 to approximately 5512 Hz (half the sampling rate), which encompasses the frequency range of interest for the vowels under analysis. In Figure 1a we show an example of a log power spectral function, i.e. log(|H(f)|^2), where H is the frequency response computed in (1), extracted from an isolated audio. We also show the associated spectrogram, reflecting the spotlight of resonant frequencies characteristics through the spectral profile representation. In addition to the analysis conducted using an LPC order of 13, we performed a sensitivity analysis by calculating spectral profiles for a range of LPC orders, from 10 to 20. This exploration aimed to evaluate the robustness of the method and the sensitivity of the corresponding analysis to the chosen LPC order.

### Selection of representative spectra

To proceed with the analysis, we identified the most representative spectral profiles for each speaker and vowel. This was accomplished through a community detection approach based on spectral similarity. Initially, we calculated the pairwise correlations between the log power spectral functions. These correlations were then binarized using a threshold of 0.9, resulting in an undirected adjacency matrix. We utilized the BCT toolbox [Rubinov & Sporns, 2010] to perform community detection on this matrix, resulting in a separation of the spectral functions into distinct communities. From this, we selected the largest community, also known as the giant component, which we considered to represent the most characteristic spectral patterns for each subject and vowel. The spectral profiles associated with this largest component were about 83% of the total and were used for all subsequent analyses.

While the primary results were obtained using a correlation threshold of 0.9, we further tested the robustness of our method by repeating the entire process with threshold values ranging from 0.5 to 0.95, in increments of 0.05. This approach allowed us to assess the stability of our results across different threshold choices and evaluate the sensitivity of the spectral functions labeled as equivalent. As a result, we demonstrated the method's potential for generalization, even in the presence of spectral variance caused by different sources of degradation.

### Visibility graphs and graph-based features

We constructed the visibility graphs from spectral profiles descriptive below. For this purpose, we used the definition of natural visibility graphs [Lacasa et al., 2008] according to equation (2). This involved transforming the spectral representations of each vowel segment into visibility graphs, where nodes represent the spectral profile amplitude of discrete frequency components and edges denote visibility relationships between nodes according to this amplitude. We show the visibility links computed between discrete frequencies of the series in the spectral domain for the previous audio example (Figure 1b) and the resulting visibility graph with a forced-based layout (Figure 1c). The construction of visibility graphs allowed us to abstract the spectral data into graphical structures, facilitating the analysis of connectivity patterns and topological features embedded within the frequency spectra. We applied a divide and conquer strategy for faster algorithm procedures [Lan et al., 2015] which has shown the fastest offline computing algorithm for general time series [Yela et al., 2020].

\begin{equation}
y_{c} < y_{b} + (y_{a}-y_{b})\frac{t_{b}-t_{c}}{t_{b}-t_{a}}
\label{eq2}
\end{equation}

Once the visibility graphs were constructed, we applied graph theory metrics to quantify various topological properties of the graphs. The metrics included were link density, average path length, clustering coefficient, and modularity. By computing these metrics for each vowel segment, we generated attribute vectors encapsulating the topological characteristics of the spectral data. These attribute vectors served as input features for subsequent speaker recognition analysis.


### Train and Test

The acoustic data segments for each subject and vowel were randomly divided into training (40%), validation (30%), and test (30%) sets. For this data split, we combined the metrics associated with the five vowels, randomly selecting segments of each vowel while ensuring that the combinations were made within each set to prevent data leakage. Thus, for each combination, we obtained a feature array containing the graph-based metrics from one example audio segment for each vowel, which could then be assigned to a corresponding subject label. We performed a total of 1000 combinations per subject for each data set, ensuring that this value was lower than the total number of possible combinations based on the available segments. These combined feature arrays were then used as inputs to the model throughout the different stages of the process.

## Models

To train and evaluate our speaker recognition model, we performed 10 runs with different random splits of training and test datasets. For the model, we employed an ensemble of decision trees using the Random Forest algorithm. The attribute vectors derived from the graph-theory metrics served as input features, while the target labels corresponded to the identities of the speakers. We trained the models using a supervised learning framework. We optimized the model hyperparameters through the maximization of the accuracy in the validation set, to reduce overfitting. We performed a grid search of hyperparameters across variations in the number of estimators (n_estimators) ranging between 5 and 50 with steps of 5, and the maximum depth of the trees (max_depth) ranging between 5 and 15 with steps of 1. We selected optimal parameters based on the validation scores obtained during the grid search. Then models were trained for this fine-tuned hyperparameters across the development set data (train and validation). To assess the performance of our speaker recognition model, we used standard performance metrics such as precision, recall, and F1-score. Additionally, we performed feature importance analysis using Random Forest models to identify the most discriminative topological properties for speaker recognition. We computed Shapley values using SHapley Additive exPlanation (SHAP) [Lundberg et al., 2017] with an efficient tree-based implementation [Lundberg et al., 2020] on the test set. The interventional method was applied to break feature correlations, ensuring a more accurate estimation of each attribute’s contribution [Janzing et al., 2020]. This analysis provided insights into the spectral patterns captured by visibility graphs, reinforcing their utility for speaker identification.


## Repository Structure

### Data

- 'variables/vg_metrics/': This directory contains 

a subdirectory for each individual along with their corresponding audio files.

### Code
