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

Ensure you have a pre-trained model saved in models/bert_model/.

Run the Streamlit dashboard:

streamlit run app.py

Interact via UI:

Enter text for single prediction

Upload CSV for bulk predictions

Project Structure
toxicity_detection/
├─ app.py
├─ model_utils.py
├─ data_utils.py
├─ train_model.py
├─ models/
├─ requirements.txt
├─ README.md
└─ sample_data/