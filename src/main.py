import pandas as pd
from data_util.data_util import *
from model.AttentionModel import *
from sklearn.metrics import precision_recall_curve, roc_curve, accuracy_score, auc, precision_score, recall_score, f1_score
from visual.PlotAUC import *
import argparse

parser = argparse.ArgumentParser(description='Run MIL Model')
parser.add_argument('-d', '--dataset', help='dataset path', default='')
parser.add_argument('-f', '--cv_fold', help='number of cross validation fold', default=5)
parser.add_argument('-ts', '--testset_size', help='Size of Testset', default=0.2)
parser.add_argument('-us', '--used_saved_split', help='Use predefined split of SUBJECTID identify by cross validation', default=False)
parser.add_argument('-s', '--save_split_path', help='Path of predfine splits', default='./splits.pkl')
parser.add_argument('-mf', '--save_model_path',  help='Path to save trained model', default='./save_model')
parser.add_argument('-mr', '--save_model_result_path',  help='Path to save model performance on testset', default='./model_results')
parser.add_argument('-ne', '--num_epochs', help='number of training epochs', default=100)
parser.add_argument('-lr', '--learning_rate', help='learning rate use for training', default=0.000402)
args = vars(parser.parse_args())


def milModel(trainData, testData, model_name, epochs, lr):
    train_set = trainData[0]
    train_label = trainData[1]
    # getting the mdoel
    model = trainModel(train_set, train_label, epochs, lr=lr, 
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

    # get the dataset 
    mainData = DataTable(**args)

    # run cross validation using MIL 
    r = result()
    r.k = counter = 0
    for trainData, testData in mainData:
        model_path = "{}/{}.pth".format(args['save_model_path'], str(counter))
        test_label, Y_prob = milModel(trainData, testData, model_path, int(args["num_epochs"]), float(args['learning_rate']))
        r.add_result(test_label, Y_prob, "MIL")  
        counter += 1
        r.k = counter

    # save model result
    t = open("{}/result.pkl".format(args['save_model_result_path']), "wb")
    pickle.dump(r, t)

    # plot AUC curve
    ROC = []
    precRec = []
    for x in range(counter):
        probArray = r.result['MIL']['prob'][str(x)]
        test_label = r.result['MIL']["label"][str(x)]

        pre, rec, thr = precision_recall_curve(test_label, probArray, pos_label=1)
        precRec.append([pre, rec, thr])
        roc_fpr, roc_tpr, thr = roc_curve(test_label, probArray, pos_label=1)
        ROC.append([roc_fpr, roc_tpr, thr])

    plotROC(ROC)
    plotPrecRec(precRec)