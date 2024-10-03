## Hierarchical Label Prediction for VNExpress Articles
## Project Overview

This project builds a machine learning model to predict hierarchical labels for articles from VNExpress. The model is designed to classify articles into categories by leveraging both text embeddings from BERT and the hierarchical relationships between labels using a Graph Convolutional Network (GCN).

## Features

- **Data Processing**: Cleaned and preprocessed Vietnamese text data using named entity recognition (NER) and removed stop words to prepare articles for model input.
- **Model Design**: Constructed a model using a BERT encoder for text representation and a GCN to model hierarchical label dependencies. The architecture also integrates dropout layers and linear layers for final classification.
- **Model Training**: Trained the model using gradient scaling and early stopping techniques to avoid overfitting. The performance was evaluated using accuracy, F1-score, precision, and recall metrics.


## Installation
1. Clone this repository:

    ```bash
    git clone https://github.com/hfakeeeeee/Hierarchical-Text-Classification-integrating-Knowledge-Graph.git
    ```

2. Navigate to the project directory:
    ```bash
    cd /Hierarchical-Text-Classification-integrating-Knowledge-Graph
    ```

3. Install the required dependencies:
    ```bash 
    pip install -r requirements.txt
    ```

## Dataset
The dataset consists of Vietnamese news articles from VNExpress, each labeled with hierarchical categories.

## Training
### Training involves:

- Early stopping to prevent overfitting.
- Gradient scaling to optimize large model training.
- Metrics such as accuracy, F1-score, precision, and recall to evaluate the performance.

### Evaluation
The model was evaluated using a test set with the following metrics:

- Accuracy: Measures the overall correctness of predictions.
- F1-Score: Balances precision and recall, especially important in hierarchical classification.
- Precision & Recall: Evaluates the model’s ability to correctly classify positive cases and retrieve all relevant instances.

### Results
After training, the model achieved satisfactory performance on the test set. Detailed metrics and visualizations of the results can be found in the report.

## How to Use
Run the preprocessing step to process the dataset before training:

```bash
python preprocess.py
```

Train and evaluate the model:

```bash
python train.py
```

Modify the configurations in config.py as needed to adjust training parameters like batch size, learning rate, and number of epochs.