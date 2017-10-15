'''
ROC Curve Plotting
Author: Yuya Jeremy Ong (yjo5006@psu.edu)
'''
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

''' Load CSV Data '''
bl119_xg = pd.read_csv('data/baseline120_xgboost_classifier_AVG_KF-10.csv')
bl119_rf = pd.read_csv('data/baseline120_random_forest_classifier_AVG_KF-10.csv')
bl119_nb = pd.read_csv('data/baseline120_naive_bayes_AVG_KF-10.csv')

plt.plot(bl119_xg.ix[:,0], bl119_xg.ix[:,1], lw=1, label='BL119-XG')
plt.plot(bl119_rf.ix[:,0], bl119_rf.ix[:,1], lw=1, label='BL119-RF')
plt.plot(bl119_nb.ix[:,0], bl119_nb.ix[:,1], lw=1, label='BL119-NB')

print(auc(bl119_xg.ix[:,0], bl119_xg.ix[:,1]))
print(auc(bl119_rf.ix[:,0], bl119_rf.ix[:,1]))
print(auc(bl119_nb.ix[:,0], bl119_nb.ix[:,1]))

''' Plot Curve '''
plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Random')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('AUC Comparsion of Top Performing Models')
plt.legend(loc="lower right")
plt.show()
