# Bag-of-Words + Support Vector Machine

## Description
Model based on feature extraction of headlines using Bag-of-Words vectorizer, trained with naive bayes algorithm.

## Preprocessing
* Utilizes standard NTLK library for tokenization process.

## Model Architecture
Bag of Words Vectorizer (from Scikit-Learn):
* `MAX_FEATURES`: The number of dimension for the word embedding model. [10, 5000]

Random Forest Classifier (from Scikit-Learn):
* `kernel`: Utilized both `linear` and `rbf`.

## Training & Evaluation Process
* Employed 10-Fold Cross Validation for Model Evaluation

## Hyperparameter Evaluation
We evaluated our model based on evaluating the dimensionality of the BoW Feature.

### SVM-Linear Model
| DIM | ACC | PRE | REC | F1 | AUC | KAPP |
| --- | --- | --- | --- | --- | --- | --- |
|  10 | 0.347594235012 | 0.253141702582 | 0.814615474014 | 0.385744902498 | 0.502406554467 | 0.00260265799232 |
|  25 | 0.540580508757 | 0.255796644481 | 0.426336141657 | 0.316035521112 | 0.503243143521 | 0.00594468103686 |
|  50 | 0.515955873936 | 0.255045856583 | 0.476636083697 | 0.33039315423 | 0.50303919629 | 0.00530362595734 |
| 100 |
| 150 |
| 200 |
| 300 |
| 400 |
| 500 |
| 1000 |
