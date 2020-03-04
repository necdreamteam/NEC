
import pandas as pd
from data_util.data_util import *
from AttentionModel.model import Attention
import torch.optim as optim
from torch.autograd import Variable
import torch
import torch.utils.data as data_utils
import torch.optim as optim
from sklearn.metrics import precision_recall_curve, roc_curve, accuracy_score, auc, precision_score, recall_score, f1_score
from visual.PlotAUC import *


def milModel(trainData, testData, model_name):
    train_set = trainData[0]
    train_label = trainData[1]
    # getting the mdoel
    model = trainModel(train_set, train_label, 50, lr=0.000402, 
                                        betas=(0.9, 0.999), 
                                        decay=0.000376, 
                                        fn = 256, 
                                        sn = 128, 
                                        tn = 64,
                                        dp = 0.2)
    torch.save(model.state_dict(), model_name)

    test_set = testData[0]
    test_label = testData[1]

    # get the probabablity dataset
    attenArray, probArray, predictArray = testModel(model, test_set, test_label)

    pre, rec, thr = precision_recall_curve(test_label, probArray, pos_label=1)
    roc_fpr, roc_tpr, thr = roc_curve(test_label, probArray, pos_label=1)
    print(auc(roc_fpr, roc_tpr))

    return test_label, probArray


class result:
    def __init__(self):
        self.result = dict()
        self.k = None
    
    def add_result(self, test_label, Y_prob, clf):
        if clf not in self.result.keys():
            self.result[clf] = dict()
            self.result[clf]['label'] = dict()
            self.result[clf]['prob'] = dict()
        
        key = "{}".format(self.k)
        self.result[clf]['label'][key] = test_label
        self.result[clf]['prob'][key] = Y_prob



if __name__ == "__main__":
    mainData = DataTable(csv_path="ml_final.csv")
    r = result()
    r.k = counter = 0
    for trainData, testData in mainData:
        test_label, Y_prob = milModel(trainData, testData, "mil_model_no_dol/{}.pth".format(str(counter)))
        r.add_result(test_label, Y_prob, "MIL")  
        counter += 1
        r.k = counter

    t = open("mil_model/result.pkl", "wb")
    pickle.dump(r, t)