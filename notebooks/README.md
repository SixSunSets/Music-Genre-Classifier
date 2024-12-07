# Music Genre Classifier with Machine Learning

## Table of Contents
1. [Overview](#overview)
2. [MFCCs: Mel-Frequency Cepstral Coefficients](#mfccs-mel-frequency-cepstral-coefficients)
3. [Data Preprocessing](#data-preprocessing)
5. [Model Selection and Comparison](#model-selection-and-comparison)
6. [Results and Evaluation](#results-and-evaluation)
7. [Prerequisitess](#prerequisites)


## Overview
This notebook is use to extract features from audio files and use them to train various ML models. One of the most effective features for audio classification is the Mel-frequency Cepstral Coefficients (MFCCs), which represent the spectral characteristics of the audio signal.

## Flow chart
<p align="center">
  <img src="https://github.com/user-attachments/assets/1b2828bf-eb22-48b7-a0c2-2e39338a7163" width="750"/>
</p>


## Data Preprocessing

### Dataset used
The dataset used is GTZAN. Inside the ``original_genres`` folder of the used dataset we found 1000 audio tracks in WAV format, with a duration of 30 seconds each, divided into 10 different musical genres. 

More details here: [GTZAN Dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification/data)

- **Genres:** Blues, Classical, Country, Disco, Hiphop, Jazz, Metal, Pop, Reggae, Rock
  
<p align="center">
  <img src="https://github.com/user-attachments/assets/aab54d0c-8bd3-43e0-a914-526f6c1a40bb" width="750"/>
</p>

## The key features: MFCCs 
Mel Frequency cepstral coefficients are used to extract relevant features from audio signals based on the way humans perceive frequencies [1]. In music genre analysis, MFCCs help to identify patterns in timbre and sound texture that vary by genre.

Here, we compare the MFCCs of two different audio tracks. The MFCCs provide insights into the differences in spectral content between the tracks, which are crucial for genre classification.

<p align="center">
  <img src="https://github.com/user-attachments/assets/5175e1a8-a890-40f0-9010-8e8057b745b1" width="750"/>
</p>


## Model Selection and Comparison
Several ML models are evaluated for the task, including:
- **Random Forest Classifier**
- **XGBoost Classifier**
- **K-Nearest Neighbors (KNN)**

Hyperparameter tuning is performed for each model to ensure optimal performance. The models are then evaluated using accuracy as the main evaluation metric, along with a detailed classification report.

### Results
After training and evaluating the models, the results indicate that different models perform differently depending on their characteristics.

| Model            | Accuracy   |
|------------------|------------|
| Random Forest    | 81.07%     |
| XGBoost          | 79.06%     |
| K-Nearest Neighbors | 85.97%   |

## Results and Evaluation

## Prerequisites
Python packages necessary specified in requirements.txt run:
```
pip install -r requirements.txt
```
## References

[1] Muda, L., Begam, M., & Elamvazuthi, I. (2010). Voice recognition algorithms using Mel Frequency Cepstral Coefficient (MFCC) and Dynamic Time Warping (DTW) techniques. Journal of Computing, 2(3), 138-143. Retrieved from https://arxiv.org/pdf/1003.4083

