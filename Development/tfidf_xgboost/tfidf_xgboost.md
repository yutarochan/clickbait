# Bag-of-Words + Logistic Function Model

## Description

## Preprocessing
* Utilizes standard NTLK library for tokenization process.

## Model Architecture
Bag of Words Vectorizer (from Scikit-Learn):
* `MAX_FEATURES`: The number of dimension for the word embedding model. [10, 5000]

Logistic Regression (from Scikit-Learn):
* `class_weight`: Used to balance the classes through a weighting heuristic: `n_samples / (n_classes * np.bincount(y))`
* `regularization`: Utilize '``'

## Training Process

## Results

## References
