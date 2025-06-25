# Speech-Emotion-Recognition

This project aims to classify emotions from human speech using deep learning techniques. Leveraging the RAVDESS dataset, we build a hybrid CNN-BiLSTM-Attention model that processes Log-Mel Spectrograms to detect 8 different emotions from both speech and song modalities. The final model achieves over 82% accuracy with class-wise precision exceeding 75% for most categories.

## 📁 Project Structure

```
.
├── emotion_model.h5           # Trained deep learning model
├── train_mean.npy             # Mean values from training data (for normalization)
├── train_std.npy              # Std values from training data (for normalization)
├── test_model.py              # Python script to test a new audio file
├── sample_audio/
│   └── test_audio.wav         # Example input audio
└── README.md                  # This documentation
```

---

## 📌 Dataset Used
- **Source**: [RAVDESS Dataset](https://zenodo.org/record/1188976)
- **RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song)
- Contains 24 actors speaking with **emotions** across 2 modalities: `speech` and `song`
- File naming convention: `Modality-Voice-Emotion-Intensity-Statement-Repetition-Actor.wav`
- Audio sampled at 48 kHz, processed down to 22.05 kHz

---

### Emotion Label Mapping

| Label ID | Emotion   |
|----------|-----------|
| 1        | Neutral   |
| 2        | Calm      |
| 3        | Happy     |
| 4        | Sad       |
| 5        | Angry     |
| 6        | Fear      |
| 7        | Disgust   |
| 8        | Surprise  |

---

### ⚙️ Preprocessing Pipeline (Step-by-Step)

1. **Load Audio Files**  
   - All `.wav` files from `speech` and `song` folders (Actor_01 to Actor_24) are read.

2. **Label Extraction**  
   - Emotion class (1–8) is parsed from filename (`Modality-Voice-Emotion-...`)
   - Adjusted to 0-based index (0–7) for modeling.

3. **Feature Extraction**  
   - Each file is converted to a **Log-Mel Spectrogram**:
     - Sampling Rate = 22050 Hz  
     - Duration = 3 seconds  
     - `n_mels` = 128  
     - Output shape = (128, 128)  
   - If shorter: zero-padded  
   - If longer: truncated

4. **Dataset Splitting**  
   - Stratified train-test split (80% train, 20% test)

5. **Normalization**  
   - Mean and standard deviation are computed **only on the training set**
   - Test set is normalized using these same values  
   - Formula:  
     ```
     X_normalized = (X - mean_train) / std_train
     ```

6. **Pseudo-Augmentation (Training Set Only)**  
   - Each spectrogram is randomly augmented by:
     - Adding Gaussian noise  
     **OR**
     - Masking random time slices (10% width)

7. **Data Shaping for CNN**  
   - Reshaped to 4D tensor format:  
     ```
     Input shape = (samples, 128, 128, 1)
     ```

8. **Label Encoding**  
   - Emotion classes are one-hot encoded into 8-dimensional vectors

9. **Class Weight Calculation**  
   - Class weights computed using `sklearn`  
   - Additional boost (×1.5) applied to `surprise` class to improve recall
     
---

## 🧠 Model Architecture

> A hybrid **CNN + BiLSTM + Attention** deep learning model

```text
Input: 128x128 log-mel spectrogram
→ Conv2D → MaxPooling → BatchNorm → Dropout
→ Conv2D → MaxPooling → BatchNorm → Dropout
→ Conv2D → MaxPooling → BatchNorm → Dropout
→ Reshape for sequence modeling
→ BiLSTM → Attention → Dropout
→ Dense (Softmax)
```

- Loss Function: `Categorical Crossentropy` with label smoothing
- Optimizer: `Adam`
- Custom Attention Layer to focus on temporal features

---

## 🎯 Performance Metrics

Tested on 20% held-out stratified data.

| Metric         | Score     |
|----------------|-----------|
| **Accuracy**   | 82%       |
| **Macro F1**   | 83%       |
| **Weighted F1**| 82%       |

✅ All emotion classes exceed **75% precision**.

---

## 🚀 How to Use

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/Speech-Emotion-Recognition.git
cd speech-emotion-recognition
```

### Step 2: Install Dependencies

```bash
pip install numpy librosa tensorflow
```

Or:

```bash
pip install -r requirements.txt
```

### Step 3: Run Inference on Audio File

```bash
python test_model.py sample_audio/test_audio.wav
```

🗣️ Example Output:

```
Predicted Emotion: calm (Confidence: 0.95)
```

---

## 🛠️ Notes

- Only `.wav` audio files with **≤3 seconds** are supported.
- The test script requires all 3 files:
  - `emotion_model.h5`
  - `train_mean.npy`
  - `train_std.npy`

---

## 📽️ Demo

> Video - https://drive.google.com/file/d/13WOmXFBoGCV-Mky2Qwk1spaohMHdFZac/view?usp=sharing
> Webapp - https://speech-emotion-recognition-ghzvmwjxdb9o6djduhbua2.streamlit.app/
---
