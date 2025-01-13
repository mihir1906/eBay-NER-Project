# eBayDE NER: German Listing Entity Recognition System

**eBayDE NER** is an end-to-end Named Entity Recognition (NER) system for German eBay listing titles. This project is built using Python, Pandas, Simple Transformers, and Weights & Biases (wandb) for hyperparameter optimization.

## Overview

This project leverages state-of-the-art transformer models – including BERT, RoBERTa, and XLM-RoBERTa – to extract key aspects (e.g., brand, model, size) from eBay listings in German. The system is designed to preprocess raw listing data, convert tags into the IOB format, and handle missing values, ensuring robust model performance. Through extensive hyperparameter tuning and experiment tracking with wandb, our best configuration achieved an F1 score of **0.98** (as reported during optimization). 

## Features

- **Model Fine-Tuning:**  
  Developed and fine-tuned multiple transformer models (BERT, RoBERTa, and XLM-RoBERTa) for German-language NER.
- **Data Preprocessing Pipelines:**  
  Engineered end-to-end pipelines using Python and Pandas to remove HTML tags, convert data to the IOB format, and impute missing values.
- **Experiment Tracking:**  
  Optimized hyperparameters using Weights & Biases (wandb) to achieve state-of-the-art performance.
- **Comparative Analysis:**  
  Detailed performance metrics (eval_loss, precision, recall, and F1 score) for multiple models are compared in the plots below.


## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/eBay-NER-Project.git
   cd eBay-NER-Project

2. Install Dependencies:
   ```bash
    pip install -r requirements.txt

##Usage
###Running the NER Model
The main inference script is located in test_ner.py. This script loads the best model and performs NER on an input tagline.
In order to use the best model unzip the model into and folder and update the path in the test_ner.py. 

1. Execute the Script:
   ```bash
   python3 src/test_ner.py

After executing you'll be promted to enter a tagline for NER.

Example:

Enter a tagline to perform NER on:
"Puma Laufschuhe Größe 42"
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 1878.33it/s]
Running Prediction: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  3.80it/s]
Sentence: "Puma Laufschuhe Größe 42"
Predictions: [{'Puma': 'B-No Tag'}, {'Laufschuhe': 'B-Produktart'}, {'Größe': 'B-No Tag'}, {'42"': 'B-EU-Schuhgröße'}]


## Comparative Analysis

Below is a combined comparative analysis of the models:

![Comparative Analysis](outputs/plots/metric_plots/f1_score_comparison.png)


  


