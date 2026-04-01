# Toxicity Detection Project

## Overview
A deep learning-based solution to detect toxic comments with a Streamlit dashboard for real-time predictions and bulk CSV predictions.

## Features
- Clean and preprocess comments
- Train deep learning models (BERT recommended)
- Real-time prediction in Streamlit
- Bulk CSV predictions
- Sample test cases display

## Installation
1. Clone the repo:
```bash
git clone <repo_url>
cd toxicity_detection

Install dependencies:
pip install -r requirements.txt

Usage

Model Creation:

### 1. Install Dependencies
pip install pandas numpy tensorflow


### 2. Prepare Dataset
Make sure these files are present in your project folder:
- `train.csv`
- `test.csv`

### 3. Run the Script
python train.py

### 4. Output Files
After execution, the following files will be generated:
- `cnn_toxic_model.h5` → Trained CNN model
- `tokenizer.pkl` → Saved tokenizer

Ensure you have a pre-trained model saved in cnn_toxic_model.h5.

Run the Streamlit dashboard:

streamlit run app.py

Interact via UI:

Enter text for single prediction

Upload CSV for bulk predictions

Project Structure
toxicity_detection/
├─ app.py
├─ preprocess.py
├─ cnn_toxic_model.h5
├─ tokenizer.pkl
├─ requirements.txt
├─ README.md
