# Bag-of-Words + Random Forest Classifier Model

## Description
Model based on feature extraction of headlines using Bag-of-Words vectorizer, trained with naive bayes algorithm.

## Preprocessing
* Utilizes standard NTLK library for tokenization process.

## Model Architecture
Bag of Words Vectorizer (from Scikit-Learn):
* `MAX_FEATURES`: The number of dimension for the word embedding model. [10, 5000]

Random Forest Classifier (from Scikit-Learn):
* `criterion`: Utilize an entropy based criterion for evaluating the model.

## Training & Evaluation Process
* Employed 10-Fold Cross Validation for Model Evaluation

## Hyperparameter Evaluation
We evaluated our model based on evaluating the dimensionality of the BoW Feature.

| DIM | ACC | PRE | REC | F1 | AUC | KAPP |
| --- | --- | --- | --- | --- | --- | --- |
|  10 | 0.740571195367 | 0.282949906443 | 0.0181315346888 | 0.0340107878351 | 0.501148327786 | 0.00339886096875 |
|  25 | 0.721004507293 | 0.25092907308 | 0.0543316545876 | 0.0891688637642 | 0.50009171065 | 0.000213250393171 |
|  50 | 0.710425110968 | 0.262857370904 | 0.0818709392574 | 0.124647327399 | 0.502138342891 | 0.00552517979627 |
| 100 | 0.703144239183 | 0.276461754992 | 0.109881076502 | 0.156939118042 | 0.506619356446 | 0.0164818700811 |
| 150 | 0.708376941341 | 0.289516691557 | 0.107992866431 | 0.156997070394 | 0.509432017326 | 0.023748340987 |
| 200 | 0.701153502126 | 0.259230125207 | 0.100802476271 | 0.144873678667 | 0.502206724274 | 0.00542091483819 |
| 300 | 0.703997998915 | 0.265194142524 | 0.0976380882729 | 0.142472711362 | 0.503041572019 | 0.00771706434946 |
| 400 | 0.709685601952 | 0.264991719982 | 0.0852917201128 | 0.128848486172 | 0.502754638051 | 0.00710003743477 |
| 500 | 0.710481832099 | 0.264080338426 | 0.0830838326023 | 0.126209143606 | 0.502536120095 | 0.00659060259263 |
| 1000 | 0.717250402772 | 0.270354617616 | 0.071436583193 | 0.112933316733 | 0.503193982344 | 0.00844803740085 |
