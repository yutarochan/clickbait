# Models
The following file tracks all of the scores of the models we have developed to
keep track of overall progress and comparison metrics.

## Headline Models
Results of models which make only use of the headlines as only source of features.

| Model ID | Accuracy | Precision | Recall | F-Measure | AUC | Kappa |
|---|---|---|---|---|---|---|
| baseline | 0.712530098741 | 0.2748983672 | 0.0814621814088 | 0.123106218277 | 0.503539076042 | 0.00926732275688 |
| bow-150_logistic | 0.522722989391 | 0.255192352432 | 0.46522399673 | 0.329347956167 | 0.503685863412 | 0.00579905315566 |
| bow-400_nb | 0.408966951048 | 0.255568595277 | 0.701283782121 | 0.373977212484 | 0.506139352906 | 0.00786738663522 |
| bow-150_randforest |  0.708376941341 | 0.289516691557 | 0.107992866431 | 0.156997070394 | 0.509432017326 | 0.023748340987 |
| bow_svm | N/A | N/A | N/A | N/A | N/A | N/A |
