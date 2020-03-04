import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np

# sklearn package for preprocess and spliting data
from sklearn.metrics import precision_recall_curve, auc # for roc curve
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
import pickle

class DataTable:
    def __init__(self, csv_path = None, k_fold = 5, testSet = False, testSetSize = 0.20, keep_split_data = False, split_data="split.pkl"):
        # import that data 
        if csv_path:
            self.csv_path = csv_path
        else:
            self.csv_path = "../reducedFeatureDataSet.csv"
        
        # get the whole test set
        self.dataset = self.importDataSet(self.csv_path)

        self.testSetSize = testSetSize
        
        # set aside some data for testing. 
        self.testSetSubjects = None 
        if testSet:
            self.testSetSubjects = self.setAsideTestSet(self.dataset, testSetSize)
        
        if keep_split_data:
            self.splitedSubjectes = self.splitData(self.dataset, self.testSetSubjects, k_fold = k_fold)
            pickle_out = open(split_data,"wb")
            pickle.dump(self.splitedSubjectes, pickle_out)
            pickle_out.close()
        else:
            pickle_in = open(split_data,"rb")
            self.splitedSubjectes = pickle.load(pickle_in)


        self.index = 0

    def __iter__(self):
        return self
    
    def __next__(self):
        try:
            # get the subjects
            train_sub, test_sub = self.splitedSubjectes[self.index]

            # training dataset we perform normaliztion and oversampling
            train_data = self.getDatasetBySubjects(self.dataset, train_sub)
            train_data = self.oversample(train_data)
            train_data = self.normalizeDataset(train_data)

            # test set only need to be normalized 
            test_data = self.getDatasetBySubjects(self.dataset, test_sub)
            test_data = self.normalizeDataset(test_data)

            train_set = self.covertToBags(train_data)
            test_set = self.covertToBags(test_data)        
        
        except IndexError:
            raise StopIteration

        self.index += 1
        return [train_set, test_set]
    
    def setAsideTestSet(self, dataset, testSize):
        nonaffected_subjects = dataset.loc[dataset.AFFECTED == "No", "SUBJECTID"].unique()
        affected_subjects = dataset.loc[dataset.AFFECTED == "Yes", "SUBJECTID"].unique()
        
        affected_subjects = np.random.choice(affected_subjects, int(len(affected_subjects) * testSize), False)
        nonaffected_subjects = np.random.choice(nonaffected_subjects, int(len(nonaffected_subjects) * testSize), False)

        return list(affected_subjects) + list(nonaffected_subjects)


    def normalizeDataset(self, dataset, col_to_drop = ["AFFECTED", "SUBJECTID"]):
        min_max_scaler = preprocessing.MaxAbsScaler()
        Cols = dataset.drop(col_to_drop, axis = 1).columns
        dataset.loc[:, Cols] = min_max_scaler.fit_transform(dataset.loc[:, Cols].values)
        return dataset

    def getDatasetBySubjects(self, dataset, subjects):
        return dataset.loc[dataset.SUBJECTID.isin(subjects), :].copy()

    def oversample(self, dataset, pos_label = "Yes"):
        oversample = dataset.loc[dataset.AFFECTED == pos_label, :].copy()
        oversample.loc[:, "SUBJECTID"] = oversample.loc[:, "SUBJECTID"] + 1000000
        dataset = pd.concat([dataset, oversample])
        return dataset

    def covertToBags(self, dataset, col_to_drop = ["AFFECTED", "SUBJECTID"]):
        Cols = dataset.drop(col_to_drop, axis = 1).columns

        data = []
        label = []
        for name in dataset.SUBJECTID.unique():
            data.append(dataset.loc[dataset.SUBJECTID == name, Cols].values)
            if dataset.loc[dataset.SUBJECTID == name, "AFFECTED"].values[0] == "Yes":
                label.append(1)
            else:
                label.append(0)
        return  [data, label]


    def importDataSet(self, csv_path):
        dataset = pd.read_csv(csv_path)
        return dataset

    '''
    SplitData seperates the dataset into Stratified K fold which preserves the
    dataset's ratio of positive and negative patient
    '''
    
    def splitData(self, dataset, ignoreSubjects, pos_label = "Yes", label_col = "AFFECTED", k_fold = 5):
        
        subjectId = list(dataset.SUBJECTID.unique())
        if (ignoreSubjects):
            for ignore in ignoreSubjects:
                subjectId.remove(ignore)
            
        subjectLabel = []
        for id in subjectId:
            if dataset.loc[dataset.SUBJECTID == id, label_col].values[0] == pos_label:
                subjectLabel.append(1)
            else:
                subjectLabel.append(0)
        
        splited_set = []
        # get the index of all splited patients
        kf = StratifiedKFold(n_splits=k_fold, shuffle=True)
        for train_index, dev_index in kf.split(subjectId, subjectLabel):
            train_set = [subjectId[index] for index in train_index]
            dev_set = [subjectId[index] for index in dev_index]
            splited_set.append([train_set, dev_set])

        return splited_set