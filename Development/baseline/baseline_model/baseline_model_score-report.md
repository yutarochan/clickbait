# Baseline Model
**Model Performance Score Report**

### K-Fold Classification Report
| K | Accuracy | Precision | Recall | F-Measure | AUC | Kappa |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.720864127345 | 0.228187919463 | 0.0829268292683 | 0.121645796064 | 0.498839248585 | -0.00298222856552 |
| 2 | 0.721274175199 | 0.367647058824 | 0.0529661016949 | 0.0925925925926 | 0.509764543849 | 0.0267848266014 |
| 3 | 0.713310580205 | 0.289156626506 | 0.110599078341 | 0.16 | 0.510737605636 | 0.027099675858 |
| 4 | 0.708759954494 | 0.324840764331 | 0.111597374179 | 0.166123778502 | 0.515060793162 | 0.0382706103339 |
| 5 | 0.705915813424 | 0.292993630573 | 0.101769911504 | 0.151067323481 | 0.508388784236 | 0.0213288768959 |
| 6 | 0.711604095563 | 0.197604790419 | 0.0812807881773 | 0.115183246073 | 0.491084181071 | -0.0224564945568 |
| 7 | 0.704778156997 | 0.25974025974 | 0.0898876404494 | 0.133555926544 | 0.501531786714 | 0.00390837374534 |
| 8 | 0.695108077361 | 0.238095238095 | 0.0892857142857 | 0.12987012987 | 0.495787895311 | -0.0106048906049 |
| 9 | 0.722980659841 | 0.22972972973 | 0.0380313199105 | 0.0652591170825 | 0.49727652952 | -0.00751269178899 |
| 10 | 0.720705346985 | 0.320987654321 | 0.0562770562771 | 0.0957642725599 | 0.506919392336 | 0.0188371696505 |

### Average Confusion Matrix
| | Pred POS | Pred NEG |
| --- | --- | --- |
| **True POS** | 36.0 | 407.3 |
| **True NEG** | 98.1 | 1216.7 |

### Average Model Performance Metrics
| ACC | PRE | REC | F1 | AUC | KAPP |
| --- | --- | --- | --- | --- | --- |
| 0.712530098741 | 0.2748983672 | 0.0814621814088 | 0.123106218277 | 0.503539076042 | 0.00926732275688 |

### AUC/ROC Plot
![ROC Plot](baseline_model_auc-plot.png)