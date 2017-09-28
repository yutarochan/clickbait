# Literature Review

#### Learning to Identify Ambiguous and Misleading News Headlines
* Key Features: Ambiguity and Congruency
* Use data mining techniques to discover latent structures of misleading content.
* Congruence: average similarity score of the body to headline metric.
* Sentiment seems to also play some role in headlines.
* RTE: Recognizing Text Entailment - a text T entails another meaning of H if H
can be inferred from the meaning of T.
* Co-Training: Semi-Supervised method of using unlabeled dataset to improve
classification results. Use p=10, n=20 (try various scores and check F-Score).

#### Shallow Reading with Deep Learning: Predicting Popularity of Online Content Using Only Its Title
* Use of BiLSTM architecture with late fusion of hidden vectors.
* Use pre-trained vectors from (Wikipedia and Media Content) - try for both raw and pre-trained models.
* Baseline Models: Bag of Words + SVM, CNN + Word Embeddings.

#### Clickbait Detection
* Good background information on what constitutes clickbait (emotion, curiosity gap, )
* Bag of Words - Best Text Features
* Archetype patterns and template mapping seems to work decently well.
* Basic Features: ~80% AUC (0.79 AUC, 076 PRE, 0.76 REC) for best result.

#### Stop Clickbait: Detecting and Preventing Clickbaits in Online News Media
* Describes various key hand-crafted linguistic features which are used in models:
* Sentence Structure:
    * Length of headline
    * Average Word Length
    * Ratio of Stop Words to Number of Content Words
    * Longest Separation between Syntactically Dependent Words of Headline
* Word Patterns:
    * Presence of Cardinal Numbers in Headline
    * Presence of Unusual Punctuation Patterns
    * Number of Contracted Words
* Clickbait Language:
    * Hyperbolic Words
    * Common Clickbait Phrases
    * Internet Slangs
    * Determiners
    * Used Multinominal NB for Predicting Popularity of CB Terms
* N-Gram Features
    * Word N-Grams
    * POS N-Grams
    * Syntactic N-Grams
    * N-Gram Features grows linearly to the size of the dataset.
    * Prune Features by APRIORI-like algorithm.
* SVM + RBF Kernel Performs the Best

#### We used Neural Networks to Detect CLickbaits: You won't believe what happened next
* Use of Distributed Word Embeddings and Character Embeddings
* Model:
    * Embedding Layer: Transforms each word into embedded features.
    * Hidden Layer: Consists of Bi-Directional RNN.
    * Output Layer: Sigmoid output node that is used as the classifier 
* BiLSTM(CE+WE): 0.98 AUC, 0.98 PRE, 0.98 REC (Best Results)