# ClickBait Detection
#### You won't believe what this group in DS 310 did to get an insanely high F-Score!

## Score Summary (Sorted by AUC)

We consider `baseline120-RF` our best as XGboost may have potential to overfit the data.

We also have a voting classifier based model based on the top two results, however execution time takes too long for the cluster to run (ran out of time...) - however source code is available for anyone interested in executing our results.

| Model ID            | Accuracy     | Precision    | Recall       | F-Measure    | AUC          | Kappa         |
|---------------------|--------------|--------------|--------------|--------------|--------------|---------------|
| **baseline120-XG**      | 0.9997714938 | 0.9997772829 | 0.9993313514 | 0.99955382   | 0.9996272733 | 0.9994002399  |
| **baseline120 - RF**    | 0.9789781176 | 0.9692606655 | 0.9480447622 | 0.9585074523 | 0.968838403  | 0.9444332556  |
| baseline120 - NB    | 0.7408315408 | 0.4742799648 | 0.1518523306 | 0.2258273987 | 0.5477784605 | 0.1211913052  |
| bow-150_randforest  | 0.7083769413 | 0.2895166916 | 0.1079928664 | 0.1569970704 | 0.5094320173 | 0.023748341   |
| bow-400_nb          | 0.408966951  | 0.2555685953 | 0.7012837821 | 0.3739772125 | 0.5061393529 | 0.0078673866  |
| bow-100_svm         | 0.5805688088 | 0.2572947888 | 0.351475974  | 0.2963501091 | 0.5047821502 | 0.0084513044  |
| baseline + SKF      | 0.7137845243 | 0.2784295252 | 0.0825529254 | 0.1244736754 | 0.5045795282 | 0.0119567655  |
| tf-idf_logistic     | 0.5847219662 | 0.2562591923 | 0.3405603294 | 0.2922417317 | 0.5038181446 | 0.0067101482  |
| bow-150_logistic    | 0.5227229894 | 0.2551923524 | 0.4652239967 | 0.3293479562 | 0.5036858634 | 0.0057990532  |
| d2v-ha-400_logistic | 0.3556705285 | 0.2531872155 | 0.7978873604 | 0.3833660381 | 0.5020313264 | 0.002521641   |
| d2v-100_logistic    | 0.6183348953 | 0.2521405113 | 0.2680773006 | 0.2412830561 | 0.5020050895 | 0.0036234866  |
| tf-idf_nb           | 0.4990044698 | 0.2531936498 | 0.5060632119 | 0.3373458101 | 0.5013267089 | 0.0020879979  |
| tf-idf_xgboost      | 0.7472829156 | 0.05         | 0.000453629  | 0.0008981945 | 0.4997708546 | -0.0006828701 |
| d2v-ha-350_logistic | 0.3093136161 | 0.2511665369 | 0.8778306558 | 0.3901459543 | 0.4979274871 | -0.0024027758 |


## Repository Setup
1. Create new folder, `Data`, under the root directory of the repository.
2. Download dataset from DS 310 website and recursively unzip all files such
that it follows the following structure.

        /Data
            |- dataset/
            |- dataset_no_figures/
            |- sample_train.arff

## Target Work Schedule
The following schedule is a rough estimate of what we should be doing.
There may or may not be deviances from this depending on how things go.
This is just a rough estimate:

| Dates       | Task                                                                                    |
|-------------|-----------------------------------------------------------------------------------------|
| 9/22 - 9/24 | Literature Review and EDA + Feature Exploration                                         |
| 9/25 - 9/29 | Individually construct various models based on literature and other available features. |
| 9/30 - 10/2 | Build Ensemble Models                                                                   |
| 10/4        | All modeling should be done at this point.                                              |
| 10/4 - 10/8 | Complete paper write up.                                                                |
