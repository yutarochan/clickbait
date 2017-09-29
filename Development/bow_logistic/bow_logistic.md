# Bag-of-Words + Logistic Function Model

## Description
Model based on feature extraction of headlines using Bag-of-Words vectorizer, trained with logistic regression.

## Preprocessing
* Utilizes standard NTLK library for tokenization process.

## Model Architecture
Bag of Words Vectorizer (from Scikit-Learn):
* `MAX_FEATURES`: The number of dimension for the word embedding model. [10, 5000]

Logistic Regression (from Scikit-Learn):
* `class_weight`: Used to balance the classes through a weighting heuristic: `n_samples / (n_classes * np.bincount(y))`

## Training & Evaluation Process
* Employed 10-Fold Cross Validation for Model Evaluation

## Hyperparameter Evaluation
We evaluated our model based on evaluating the dimensionality of the BoW Feature vs F1-Measure.

| DIM | ACC | PRE | REC | F1 | AUC | KAPP |
| --- | --- | --- | --- | --- | --- | --- |
|  10 | 0.428531375452 | 0.252416219749 | 0.647512698524 | 0.363022527633 | 0.500762373063 | 0.000913912980656 |
|  25 | 0.49405495288 | 0.256167449434 | 0.528230519306 | 0.344790082312 | 0.505415364546 | 0.00808415275415 |
|  50 | 0.505317557486 | 0.255449786098 | 0.501431393559 | 0.338290640712 | 0.503923818007 | 0.00622893610111 |
| 100 | 0.524429150651 | 0.255596305231 | 0.463603421944 | 0.329330010208 | 0.504261957953 | 0.006625897995 |
| 150 | 0.522722989391 | 0.255192352432 | 0.46522399673 | 0.329347956167 | 0.503685863412 | 0.00579905315566 |
| 200 | 0.54029890807 | 0.257219458624 | 0.43648199317 | 0.323193311573 | 0.506071654011 | 0.00955204062612 |
| 300 | 0.537170094188 | 0.256868268711 | 0.442056602147 | 0.324692889886 | 0.505688162523 | 0.00887119564734 |
| 400 | 0.531312004377 | 0.257757169073 | 0.457318699307 | 0.329362907367 | 0.506894188891 | 0.0106778595427 |
| 500 | 0.530629830917 | 0.256219815509 | 0.452782583913 | 0.326968515258 | 0.504894812603 | 0.00769499901391 |
