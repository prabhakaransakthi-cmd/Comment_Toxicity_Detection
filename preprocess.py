
# Basic CNN Toxicity Detection Project
# Author: Trainee Developer (Intermediate Level)

import pandas as pd
import numpy as np
import re
import pickle

# Deep learning imports
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# ==============================
# 1. Load Dataset
# ==============================

# Load train and test CSV files
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

print("Train Shape:", train_df.shape)
print("Test Shape:", test_df.shape)

print(train_df.head())

# ==============================
# 2. Text Preprocessing
# ==============================

def clean_text(text):
    text = str(text).lower()   # convert to lowercase
    text = re.sub(r'[^a-zA-Z ]', '', text)  # remove special characters
    return text

train_df['clean_text'] = train_df['comment_text'].apply(clean_text)
test_df['clean_text'] = test_df['comment_text'].apply(clean_text)

# ==============================
# 3. Tokenization & Padding
# ==============================

max_words = 10000   # maximum number of words to keep
max_len = 200       # maximum sentence length

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(train_df['clean_text'])

X_train = tokenizer.texts_to_sequences(train_df['clean_text'])
X_test = tokenizer.texts_to_sequences(test_df['clean_text'])

X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)

# Target column (Assuming 'toxic' column exists)
y_train = train_df['toxic']

# ==============================
# 4. Build CNN Model
# ==============================

model = Sequential()

# Embedding layer converts words to vectors
model.add(Embedding(input_dim=max_words, output_dim=128, input_length=max_len))

# 1D CNN layer
model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))

# Take the most important feature
model.add(GlobalMaxPooling1D())

model.add(Dropout(0.5))  # reduce overfitting
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # binary classification

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

# ==============================
# 5. Train Model
# ==============================

model.fit(X_train, y_train,
          epochs=3,
          batch_size=32,
          validation_split=0.2)

# ==============================
# 6. Save Model & Tokenizer
# ==============================

model.save("cnn_toxic_model.h5")

# Save tokenizer using pickle
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("Model and tokenizer saved successfully!")
