# Feature extraction and Model Training

## Table of Contents
1. [Overview](#overview)
2. [Flow chart](#flow-chart)
3. [Data Preprocessing](#data-preprocessing)
4. [Feature Extraction](#feature-extraction)
5. [Selection of segmentation algorithm](#selection-of-segmentation-algorithm)
6. [Model Training](#model-training)
7. [Validation and Fine-Tuning](#validation-and-fine-tuning)
8. [Prerequisites](#prerequisites)
9. [References](#references)


## Overview
This Jupyter notebook is use to extract features from audio files and use them to train various ML models. One of the most effective features for audio classification is the Mel-frequency Cepstral Coefficients (MFCCs), which represent the spectral characteristics of the audio signal. The next intention is to test the model with new audio tracks, and then prepare a simple user interface to upload a track of your own.


## Flow chart
<p align="center">
  <img src="https://github.com/user-attachments/assets/1b2828bf-eb22-48b7-a0c2-2e39338a7163" width="750"/>
</p>

---

## Data Preprocessing

### Dataset used
The dataset used is GTZAN. Inside the ``original_genres`` folder of the used dataset we found 1000 audio tracks in WAV format, with a duration of 30 seconds each, divided into 10 different musical genres. 

- **Genres:** Blues, Classical, Country, Disco, Hiphop, Jazz, Metal, Pop, Reggae, Rock
  
More details here: [GTZAN Dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification/data)

Each 30-second audio track from the GTZAN dataset was segmented into 10-second intervals, tripling the dataset size. The features (MFCCs) extracted from each segment were used as input for training machine learning models [1].

<p align="center">
  <img src="https://github.com/user-attachments/assets/2a005970-5b98-45ad-b443-1f8929fe47be" width="750"/>
</p>

---

## Feature Extraction
The Mel-Frequency Cepstral Coefficients (MFCCs) were computed for each audio segment. These features summarize the spectral properties of the audio and help the classifier distinguish between genres.

### The key features: MFCCs 
Mel Frequency cepstral coefficients are used to extract relevant features from audio signals based on the way humans perceive frequencies [2]. In music genre analysis, MFCCs help to identify patterns in timbre and sound texture that vary by genre.

Here, we compare the MFCCs of two different audio tracks. The MFCCs provide insights into the differences in spectral content between the tracks, which are crucial for genre classification.

<p align="center">
  <img src="https://github.com/user-attachments/assets/a466b966-0b03-49fb-ac2d-bce3fbd00616" width="750"/>
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/6e684c49-b185-4d56-b6a1-27ff76da34f1" width="750"/>
</p>

The MFCCs were extracted using the following parameters:
- **Number of coefficients (n_mfcc)**: 20
- **Window size**: 2048 samples
- **Hop length**: 512 samples

Below is the code used to extract the MFCCs:
```python
mfcc = librosa.feature.mfcc(y=audio_segment, sr=sample_rate, n_mfcc=20)
```
---

## Selection of segmentation algorithm

Steps in Data Preparation:

- Convert MFCC strings to NumPy arrays: Each MFCC feature was stored as a string and transformed into arrays for compatibility with ML algorithms.
- Encode Genre Labels: Music genres were converted into numerical values using a LabelEncoder.
- Scale Features: StandardScaler was applied to standardize the feature distribution, improving model performance.
- Split Dataset: The data was divided into training (85%) and testing (15%) sets while preserving the class distribution.

---

## Model Training
Three distinct machine learning algorithms were implemented and evaluated for the task of music genre classification: Random Forest, XGBoost, and K-Nearest Neighbors (KNN) [3]. Each model was trained on the segmented and preprocessed dataset, with a portion held out for testing.

1. Random Forest is an ensemble learning method that builds multiple decision trees and merges their outputs to improve accuracy and control overfitting. It is effective for multi-class classification tasks like this one.

2. XGBoost is a gradient boosting framework known for its speed and performance in supervised learning tasks.

3. K-Nearest Neighbors (KNN) is a simple yet effective algorithm that classifies samples based on the majority class of their nearest neighbors.

The confusion matrices for each model visually depict the predictions versus the true labels, providing insights into model strengths and weaknesses.

---

## Validation and Fine-Tuning

### Confussion matrix for each model

The following confusion matrices illustrate the performance of each model in classifying musical genres in the test set. The cells show the number of examples correctly or incorrectly classified for each musical genre.

<p align="center">
  <img src="https://github.com/user-attachments/assets/6aff6aba-6500-428c-a3c2-dc33cc72931a" width="500"/>
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/e1d528f0-2f46-4560-ab3b-fd216b3af04f" width="500"/>
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/08bab2c0-03e5-4c29-aec6-86c88535717b" width="500"/>
</p>
As can be seen in the confusion matrices, Random Forest has a more balanced classification across genres, while XGBoost and KNN have more errors in genres such as Rock and Disco. Although Random Forest is the most accurate in this initial test, there may be room for improvement in all models with proper parameter tuning.

### Model Comparison

The following table shows the results of an initial test for Random Forest, XGBoost and K-Nearest Neighbors on the test data set. The metrics evaluated include Accuracy, Recall, F1-score and Accuracy.

| **Algorithm**      | **Precision** | **Recall** | **F1-score** | **Accuracy** |
|---------------------|---------------|------------|--------------|--------------|
| Random Forest       | 0.79          | 0.78       | 0.78         | 0.78         |
| XGBoost             | 0.75          | 0.75       | 0.75         | 0.75         |
| K-Nearest Neighbors | 0.75          | 0.74       | 0.74         | 0.74         |

It should be noted that these results correspond to an initial test using default parameters. In the next phase, hyperparameter tuning will be performed, which could significantly improve the performance of the models, especially for K-Nearest Neighbors, which is sensitive to the hyperparameter settings.

### Hyperparameter tuning
``GridSearchCV`` was used to perform hyperparameter fitting on each of the models. Using a dictionary, we defined a parameter space, evaluated multiple combinations and selected those that maximized model accuracy.

#### Random Forest (270 fits)
- ``n_estimators``: number of trees in the forest (50, 100, 200).
- ``max_depth``: maximum depth of each tree (10, 20, 30).
- ``min_samples_split``: minimum number of samples required to split a node (2, 5, 10).
- ``criterion``: division criterion (gini or entropy).

#### XGBoost (540 fits)
- ``n_estimators``: number of trees (50, 100, 200).
- ``max_depth``: maximum depth of the trees (3, 6, 10).
- ``learning_rate``: learning rate (0.01, 0.1, 0.3).
- ``subsample``: proportion of samples to use (0.8, 1.0).
- ``colsample_bytree``: ratio of columns to use per tree (0.8, 1.0).

#### KNN (80 fits)
- ``n_neighbors``: number of neighbors (3, 5, 7, 10).
- ``weights``: type of neighbor weighting ('uniform' or 'distance').
- ``metric``: type of distance metric ('euclidean' or 'manhattan').

### Confussion matrices for "best" models

<p align="center">
  <img src="https://github.com/user-attachments/assets/50b395c0-c460-4b4b-a1f6-c036f9eed255" width="500"/>
</p>

- Best hyperparameters for Random Forest: ``{'criterion': 'entropy', 'max_depth': 20, 'min_samples_split': 2, 'n_estimators': 200}``
- Accuracy in test set: ``0.8106904231625836``

<p align="center">
  <img src="https://github.com/user-attachments/assets/126c3f5e-45f0-4784-9c35-821252456700" width="500"/>
</p>

- Best hyperparameters for XGBoost: ``{'colsample_bytree': 0.8, 'learning_rate': 0.1, 'max_depth': 10, 'n_estimators': 200, 'subsample': 0.8}``
- Accuracy in test set: ``0.7906458797327395``

<p align="center">
  <img src="https://github.com/user-attachments/assets/4d0092f8-923e-4699-ac50-a98c44304acb" width="500"/>
</p>

- Best hyperparameters for KNN: ``{'metric': 'euclidean', 'n_neighbors': 3, 'weights': 'distance'}``
- Accuracy in test set: ``0.8596881959910914``

#### Conclusions

- KNN: 85.97%

    - It is the model with the highest accuracy, indicating that, with current features, the proximity between points in feature space is effective in correctly classifying most genres.
    - Looking at the confusion matrix, some confusion is noted in classes such as “rock” and “country”.

- Random Forest: 81.07%

    - Although it does not surpass KNN in accuracy, it performs quite well.
    - It has a more balanced performance across classes and seems to handle classes with less homogeneity or overlap better.
    - For example, although it does not have the highest accuracy, it handles classes such as “pop” and “classical” well compared to KNN.

- XGBoost: 79.06%

    - It has the lowest accuracy among the three, although it is still competitive.
    - It shows more confusion in certain classes (“hiphop”, “rock”, “country”) than the other two models, despite being a sophisticated algorithm.
    - In this specific case, seems less effective than Random Forest and KNN.
    
---

## Prerequisites
Create a virtual environment if desired, then to install the necessary Python packages specified in requirements.txt run:
```console
pip install -r requirements.txt
```

---

## References

[1] Chettiar, G., & Selvakumar, K. (2021). Music Genre Classification Techniques. Retrieved from https://www.researchgate.net/publication/356377974_Music_Genre_Classification_Techniques

[2] Muda, L., Begam, M., & Elamvazuthi, I. (2010). Voice recognition algorithms using Mel Frequency Cepstral Coefficient (MFCC) and Dynamic Time Warping (DTW) techniques. Retrieved from https://arxiv.org/pdf/1003.4083

[3] Islam, Md., Hasan, M., Rahim, M. A., Hasan, A., Mynuddin, M., Khandokar, I., & Islam, J. (2021). Machine Learning-Based Music Genre Classification with Pre-Processed Feature Analysis. Retrieved from https://www.researchgate.net/publication/357912712_Machine_Learning-Based_Music_Genre_Classification_with_Pre-_Processed_Feature_Analysis

