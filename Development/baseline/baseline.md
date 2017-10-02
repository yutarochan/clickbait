# Baseline Model

## Description
This model establishes a minimum baseline for the project, as well as serving as
a sample template for how we formulate our documentation process.

This model utilizes the sample features provided to us in the instructions for
the project. The features that this model utilizes are the following:
* Word Count (int)
* Average Word Length (int)
* Length of the Longest Word (int)
* Whether the headline starts with a number (0 or 1)
* Whether start with who/what/why/where/when/how (0 or 1)

## Preprocessing
* Utilizes standard NTLK library for tokenization process.

## Model Architecture
Gaussian Naive Bayes (from Scikit-Learn)

## Training & Evaluation Process
* Employed 10-Fold Cross Validation for Model Evaluation

### Baseline Model
| Model | ACC | PRE | REC | F1 | AUC | KAPP |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.712530098741 | 0.2748983672 | 0.0814621814088 | 0.123106218277 | 0.503539076042 | 0.00926732275688 |
| baseline + SKF | 0.713784524267 | 0.278429525246 | 0.0825529253859 | 0.124473675441 | 0.504579528246 | 0.0119567655317 |
| baseline + SMOTE | 0.415228524067 | 0.2510524696 | 0.667372985776 | 0.355361768116 | 0.498216044008 | -0.00237941811272 |
| baseline + SMOTE + SKF | 0.357927645729 | 0.243752856638 | 0.755571655177 | 0.355132389775 | 0.489737867914 | -0.0131279440044 |
