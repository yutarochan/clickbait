# Doc2Vec Headline + Logistic Function

## Description
Model based on word embedding modeling, trained with logistic function.

## Preprocessing
* Utilizes standard NTLK library for tokenization process.

## Model Architecture
Document Word Embedding:
Performed training of word embedding only on the training subset then applied inference function for new views.

Logistic Regression (from Scikit-Learn):
* `class_weight`: Used to balance the classes through a weighting heuristic: `n_samples / (n_classes * np.bincount(y))`

## Training & Evaluation Process
* Employed 10-Fold Cross Validation for Model Evaluation

## Hyperparameter Evaluation
We evaluated our model based on evaluating the dimensionality of the document vector feature.

### SVM-Linear Model
| DIM | ACC | PRE | REC | F1 | AUC | KAPP |
| --- | --- | --- | --- | --- | --- | --- |
| 50 | 0.255276746729 | 0.25223453237 | 0.994347819694 | 0.402257860876 | 0.500218314516 | 0.000233923277925 |
| 100 (NO SMOTE) | 0.618334895266 | 0.252140511302 | 0.268077300557 | 0.241283056113 | 0.5020050895 | 0.00362348655432 |
| 100 | 0.258063907963 | 0.252723352681 | 0.992483693815 | 0.40272267501 | 0.501462455498 | 0.00155506774962 |
| 200 | 0.634941186591 | 0.234530557068 | 0.225553498499 | 0.140639824149 | 0.498071488524 | -0.00631728734096 |
| 300 | 0.25982721075 | 0.252361723526 | 0.986009858671 | 0.401718918944 | 0.500517710995 | 0.000590566883884 |
