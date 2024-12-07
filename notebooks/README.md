# Music Genre Classifier with Machine Learning

## Table of Contents
1. [Overview](#overview)
2. [Flow chart](#flow-chart)
3. [Data Preprocessing](#data-preprocessing)
4. [Feature Extraction](#feature-extraction)
5. [Selection of segmentation algorithm](#selection-of-segmentation-algortihm)
6. [Model Training](#model-training)
7. [Validation and Fine-Tuning](#validation-and-fine-tuning)
8. [Testing and Evaluation](#testing-and-evaluation)
9. [Prerequisites](#prerequisites)
10. [References](#references)


## Overview
This notebook is use to extract features from audio files and use them to train various ML models. One of the most effective features for audio classification is the Mel-frequency Cepstral Coefficients (MFCCs), which represent the spectral characteristics of the audio signal.

## Flow chart
<p align="center">
  <img src="https://github.com/user-attachments/assets/1b2828bf-eb22-48b7-a0c2-2e39338a7163" width="750"/>
</p>

## Data Preprocessing

### Dataset used
The dataset used is GTZAN. Inside the ``original_genres`` folder of the used dataset we found 1000 audio tracks in WAV format, with a duration of 30 seconds each, divided into 10 different musical genres. 

- **Genres:** Blues, Classical, Country, Disco, Hiphop, Jazz, Metal, Pop, Reggae, Rock
  
More details here: [GTZAN Dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification/data)

Each 30-second audio track from the GTZAN dataset was segmented into 10-second intervals, tripling the dataset size. The features (MFCCs) extracted from each segment were used as input for training machine learning models. 

<p align="center">
  <img src="https://github.com/user-attachments/assets/aab54d0c-8bd3-43e0-a914-526f6c1a40bb" width="750"/>
</p>

## Feature Extraction
The Mel-Frequency Cepstral Coefficients (MFCCs) were computed for each audio segment. These features summarize the spectral properties of the audio and help the classifier distinguish between genres.

### The key features: MFCCs 
Mel Frequency cepstral coefficients are used to extract relevant features from audio signals based on the way humans perceive frequencies [1]. In music genre analysis, MFCCs help to identify patterns in timbre and sound texture that vary by genre.

Here, we compare the MFCCs of two different audio tracks. The MFCCs provide insights into the differences in spectral content between the tracks, which are crucial for genre classification.

<p align="center">
  <img src="https://github.com/user-attachments/assets/5175e1a8-a890-40f0-9010-8e8057b745b1" width="750"/>
</p>

The MFCCs were extracted using the following parameters:
- **Number of coefficients (n_mfcc)**: 20
- **Window size**: 2048 samples
- **Hop length**: 512 samples

Below is the code used to extract the MFCCs:
```python
mfcc = librosa.feature.mfcc(y=audio_segment, sr=sample_rate, n_mfcc=20)
```
## Selection of segmentation algorithm

Steps in Data Preparation:

- Convert MFCC strings to NumPy arrays: Each MFCC feature was stored as a string and transformed into arrays for compatibility with ML algorithms.
- Encode Genre Labels: Music genres were converted into numerical values using a LabelEncoder.
- Scale Features: StandardScaler was applied to standardize the feature distribution, improving model performance.
- Split Dataset: The data was divided into training (85%) and testing (15%) sets while preserving the class distribution.

## Model Training
Three distinct machine learning algorithms were implemented and evaluated for the task of music genre classification: Random Forest, XGBoost, and K-Nearest Neighbors (KNN). Each model was trained on the segmented and preprocessed dataset, with a portion held out for testing.

1. Random Forest is an ensemble learning method that builds multiple decision trees and merges their outputs to improve accuracy and control overfitting. It is effective for multi-class classification tasks like this one.

2. XGBoost is a gradient boosting framework known for its speed and performance in supervised learning tasks.

3. K-Nearest Neighbors (KNN) is a simple yet effective algorithm that classifies samples based on the majority class of their nearest neighbors.

The confusion matrices for each model visually depict the predictions versus the true labels, providing insights into model strengths and weaknesses.

## Validation and Fine-Tuning

### Model Comparison

| **Algorithm**      | **Precision** | **Recall** | **F1-score** | **Accuracy** |
|---------------------|---------------|------------|--------------|--------------|
| Random Forest       | 0.78          | 0.77       | 0.77         | 0.78         |
| XGBoost             | 0.79          | 0.79       | 0.79         | 0.79         |
| K-Nearest Neighbors | 0.73          | 0.72       | 0.72         | 0.72         |

### Confussion matrix for each model

<p align="center">
  <img src="https://github.com/user-attachments/assets/88a3cd0b-b1d5-476d-8b43-595bcc80b7c7" width="500"/>
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/1f082cdf-d22b-4d83-b627-122aeec534ca" width="500"/>
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/5ab9ded1-60fd-416a-9544-fbc52d25cb91" width="500"/>
</p>

## Prerequisites
Create a virtual environment if desired, then to install the necessary Python packages specified in requirements.txt run:
```console
pip install -r requirements.txt
```
## References

[1] Muda, L., Begam, M., & Elamvazuthi, I. (2010). Voice recognition algorithms using Mel Frequency Cepstral Coefficient (MFCC) and Dynamic Time Warping (DTW) techniques. Journal of Computing, 2(3), 138-143. Retrieved from https://arxiv.org/pdf/1003.4083

