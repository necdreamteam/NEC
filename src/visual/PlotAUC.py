from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt
from scipy import interp
from sklearn import svm, metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_curve, roc_curve, accuracy_score, auc, precision_score, recall_score, f1_score
import numpy as np
import os


def plotPrecRec(precRec, result_dir='', classifier=''):
    '''
    Function to plot precision recall curves

    :param precRec: [[pre, rec, thr]]. List of all precision-recall values.
    :param result_dir. The output directory.

    '''

    k = len(precRec)

    curves = np.zeros((k,1000))
    interpol = np.linspace(0, 1., 1000)
    aucs = np.zeros((k))

    for i,entry in enumerate(precRec):


        pre, rec, thr = entry

        curves[i] = interp(interpol, pre, rec)
        aucs[i] = auc(rec, pre)

    curves[:,0] = 1
    curves[:,-1] = 0
    pre_mean = np.mean(curves, axis=0)
    pre_std = np.std(curves, axis=0)
    rec_mean = interpol
    auc_mean = np.mean(aucs)
    auc_std = np.std(aucs)


    # plot
    plt.figure()
    plt.plot(rec_mean, pre_mean, label = u'AUC = %0.3f ± %0.3f ' % (auc_mean, auc_std))
    plt.errorbar(rec_mean[::100], pre_mean[::100], yerr=pre_std[::100], fmt='o', color='black', ecolor='black')
    plt.legend(loc='lower left')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(classifier + "Precision vs Recall")
    plt.savefig(os.path.join(result_dir, classifier + '_PrecisionRecall_CV.png'))
    plt.close()


def plotROC(ROC, result_dir='', classifier = ''):
    '''
    Function to plot precision recall curves

    :param ROC: [[fpr, tpr, thr]]. List of all ROC values.
    :param result_dir: The output directory.

    '''

    k = len(ROC)

    curves = np.zeros((k,1000))
    interpol = np.linspace(0, 1., 1000)
    aucs = np.zeros((k))

    for i,entry in enumerate(ROC):
        fpr, tpr, thr = entry
        curves[i] = interp(interpol, fpr, tpr)
        aucs[i] = auc(fpr, tpr)

    curves[:,0] = 0
    curves[:,-1] = 1
    tpr_mean = np.mean(curves, axis=0)
    tpr_std = np.std(curves, axis=0)
    fpr_mean = interpol
    auc_mean = np.mean(aucs)
    auc_std = np.std(aucs)


    # plot
    plt.figure()
    plt.plot(fpr_mean, tpr_mean, label = u'AUC = %0.3f ± %0.3f ' % (auc_mean, auc_std))

    plt.errorbar(fpr_mean[::100], tpr_mean[::100], yerr=tpr_std[::100], fmt='o', color='black', ecolor='black')
    plt.legend(loc='lower right')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(classifier + "ROC")
    plt.savefig(os.path.expanduser(os.path.join(result_dir, classifier + '_ROC_CV.png')), dpi=1000)
    plt.close()