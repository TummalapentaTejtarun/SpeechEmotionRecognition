# Speech Emotion Recognition

Welcome to the **Speech Emotion Recognition** project! This repository contains a Jupyter notebook (`SpeechEmotionRecognition.ipynb`) that implements a state-of-the-art system for classifying emotions from speech audio files. The system identifies seven distinct emotions: **Angry, Disgust, Fear, Happy, Neutral, Pleasant Surprise (ps), and Sad**, using advanced audio processing and deep learning techniques.

---

## Project Motivation

Speech emotion recognition is pivotal for enhancing human-computer interaction, with applications in:

- Virtual assistants  
- Mental health monitoring  
- Customer service automation  
- Sentiment analysis  

This project aims to:

- Develop a robust system to accurately classify emotions from speech  
- Explore and compare multiple deep learning models for performance  
- Provide a well-documented, reproducible workflow for researchers and developers  
- Highlight challenges (e.g., overfitting) and propose actionable improvements  

**Note:** The models achieve near-perfect accuracy (~100%) on the test set, which may indicate overfitting, data leakage, or a simplistic dataset. Robust validation techniques are recommended for real-world deployment.

---

## Features

- **Audio Processing:** Extracts Mel-Frequency Cepstral Coefficients (MFCCs) for feature representation  
- **Deep Learning Models:** Implements four architectures:
  - 1D Convolutional Neural Network (CNN)  
  - Long Short-Term Memory (LSTM)  
  - Bidirectional LSTM (BiLSTM)  
  - Gated Recurrent Unit (GRU)  
- **Evaluation Metrics:** Provides accuracy, confusion matrices, and detailed classification reports (precision, recall, F1-score)  
- **Visualizations:** Includes label distribution plots and confusion matrices for insightful analysis  
- **Environment Support:** Optimized for Google Colab with GPU acceleration; also compatible with local Python environments  

---

## Dataset

## Download link: https://www.kaggle.com/ejlok1/toronto-emotional-speech-set-tess 
The project utilizes a speech emotion dataset  with the following characteristics:

- **Size:** 5600 WAV audio files, balanced with 800 samples per emotion  
- **Emotions:** Angry, Disgust, Fear, Happy, Neutral, Pleasant Surprise (ps), Sad  
- **Storage:** Stored in `/content/drive/MyDrive/archive` for Google Colab users  
- **Labels:** Derived from filenames (e.g., `filename_angry.wav → angry`)  
- **Features:** Mean MFCCs extracted as the primary input for model training  
- **Data Split:** 80% training, 20% testing, ensuring balanced classes  

**Dataset Setup:**

1. Download a public dataset like RAVDESS from Kaggle  
2. Place audio files in `/content/drive/MyDrive/archive` (Colab) or update the notebook’s file path for local use  

---

## Methodology

The project follows a structured pipeline to process audio and classify emotions:

### 1. Data Loading and Preprocessing
- Loads WAV files from the specified directory  
- Extracts emotion labels from filenames and encodes them numerically  
- Splits data into 80% training and 20% testing sets with stratification  

### 2. Feature Extraction
- Computes **Mel-Frequency Cepstral Coefficients (MFCCs)** to capture timbral characteristics of speech  
- Normalizes features for consistent input to models  

### 3. Model Architectures
- **1D CNN:** Extracts spatial patterns using convolution and pooling layers  
- **LSTM:** Captures temporal dependencies in audio sequences  
- **Bidirectional LSTM:** Models both past and future contexts for enhanced sequence processing  
- **GRU:** Efficient temporal modeling with gated recurrent units  

### 4. Training Configuration
- **Loss Function:** Categorical cross-entropy  
- **Optimizer:** Adam  
- **Epochs:** 50–100 with early stopping to prevent overfitting  
- **Validation:** Evaluates performance on the test set  

### 5. Evaluation and Visualization
- Computes accuracy, confusion matrices, and classification reports  
- Visualizes label distribution to confirm dataset balance  
- Displays sample predictions  

---

## Results

Models were evaluated on a test set of 1120 samples (20% of dataset). Key findings:

- **Accuracy:** CNN, LSTM, BiLSTM, GRU → ~100% (potential overfitting or data leakage)  
- **Classification Report (Example for GRU):**
  - Precision, recall, F1-score: 1.00 for all emotions  
  - Support: 160 samples per emotion  
  - Macro and weighted averages: 1.00  
- **Visualizations:** Balanced label distribution confirmed; confusion matrices show perfect classification  
- **Sample Predictions:** Models accurately predict emotions (e.g., Actual: neutral → Predicted: neutral)  

**Caveats:**  
The near-perfect accuracy suggests overfitting or data leakage. Real-world deployment requires testing on diverse, noisy datasets and robust validation (e.g., k-fold cross-validation).

---

## Requirements

- Python 3.x  
- Jupyter Notebook or Google Colab  
- Libraries: `numpy`, `pandas`, `librosa`, `tensorflow`, `scikit-learn`, `matplotlib`, `seaborn`

---

## How to Run

1. Clone the repository:  
```bash
git clone <[repository_url](https://github.com/TummalapentaTejtarun/SpeechEmotionRecognition.git)>
