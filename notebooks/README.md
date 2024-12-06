# Music Genre Classifier with Machine Learning

## Table of Contents
1. [Overview](#overview)
2. [MFCCs: Mel-Frequency Cepstral Coefficients](#mfccs-mel-frequency-cepstral-coefficients)
3. [Dataset Used](#dataset-used)
4. [Model Selection and Comparison](#model-selection-and-comparison)
5. [Results and Evaluation](#results-and-evaluation)
6. [Requirements](#requirements)


## Overview
This project aims to classify music into different genres using Machine Learning (ML) techniques. The main idea is to extract features from audio files and use them to train various ML models. One of the most effective features for audio classification is the Mel-frequency Cepstral Coefficients (MFCCs), which represent the spectral characteristics of the audio signal.

## MFCCs: Mel-Frequency Cepstral Coefficients
*Mel-frequency Cepstral Coefficients* (*MFCCs*) are a compact and effective representation of the frequency spectrum of an audio signal. MFCCs are widely used in audio signal processing, particularly for tasks like speech recognition and music classification. By converting audio signals into MFCCs, we can capture the essential features that allow us to distinguish between different genres of music.

### MFCCs for Two Different Audio Tracks

Here, we compare the MFCCs of two different audio tracks. The MFCCs provide insights into the differences in spectral content between the tracks, which are crucial for genre classification.

<p align="center">
  <img src="https://github.com/user-attachments/assets/5175e1a8-a890-40f0-9010-8e8057b745b1" width="750"/>
</p>

## Dataset Used
For this project, the GTZAN dataset is used, which contains 1000 audio tracks, each 30 seconds long, and divided into 10 different music genres. The dataset is widely used for music classification tasks.

More details here: [GTZAN Dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification/data)

- **Genres:** Blues, Classical, Country, Disco, Hiphop, Jazz, Metal, Pop, Reggae, Rock
- **Audio Format:** WAV
- **Duration per track:** 30 seconds
- **Sampling Rate:** 22,050 Hz

<p align="center">
  <img src="https://github.com/user-attachments/assets/aab54d0c-8bd3-43e0-a914-526f6c1a40bb" width="750"/>
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

