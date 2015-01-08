import numpy as np
import pandas
import sys
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, roc_curve
from new_preprocess import PreProcessor
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.grid_search import GridSearchCV
from utils import plot_coeff, plot_roc_curve


class LearningFromData:
    """ A very generic class for testing different machine learning algorithms """
    """ o Gaussian Naive Bayes """
    """ o Logisitc Regression """
    """ o Random Forest """
    """ o Support Vector Classifier """
    """ o KNeighborsClassifier """ 

    def __init__(self):
        self.datamngr = PreProcessor(fillnan_method="mean")
        self.roc_ax = None

    def get_data(self):
        """ Get the data from the preprocessor """
        self.train_data, self.test_data, self.train_target, self.test_target, self.oos_data = self.datamngr.process_data()

    def prepare_service(self):
        """ Prepare some common parts for plotting """
        self.roc_fig = plt.figure()
        self.roc_ax = self.roc_fig.add_subplot(111)


    def compute_model_pipeline(self, clf, param_grid, n_jobs=1, clfid=""):
        """ The generic modeling pipeline which does cross validataion and finds the best estimator """ 
        print "Processing Model Pipeline:", clfid
        estimator = GridSearchCV(clf,  param_grid=param_grid, 
                verbose=0, n_jobs=n_jobs) 
        estimator.fit(self.train_data,  self.train_target) 
        clf = estimator.best_estimator_
        self.compute_model(clf, clfid=clfid)

    def compute_model(self, clf, clfid=""):
        """ Pass a generic model, compute classification metrics, plot the roc curve """
        print "Processing Model:", clfid
        clf.fit(self.train_data,  self.train_target) 
        pred = clf.predict( self.test_data)
        pred_prob = clf.predict_proba(self.test_data)
        print classification_report( self.test_target, pred)
        print "Accuracy", accuracy_score(self.test_target, pred)
        print "AUC prob", roc_auc_score(self.test_target, pred_prob[:,1])
        fpr, tpr, thresholds = roc_curve(self.test_target, pred_prob[:,1],  pos_label=1)
        plot_fig = True 
        if plot_fig:
            plot_roc_curve(fpr, tpr, thresholds, label=clfid, ax=self.roc_ax)

    def compute_logistic_model(self):
        """ The Logistic model, the various tuning parmeters to be set here """
        param_grid = { "penalty" :["l1","l2"] ,
                       "C": [.1, 1, 10, 100],
                      }
        logit_clf = linear_model.LogisticRegression(dual=False)
        self.compute_model_pipeline(logit_clf, param_grid, clfid="logit")

    def compute_randomforest_model(self):
        """ Random Forest Classifier """
        rf = RandomForestClassifier()
        param_grid = { "n_estimators" : [ 50, 100, 200],
                       "max_depth": [2, 4, 8, None],
                      }
        self.compute_model_pipeline(rf, param_grid, clfid="RF")

    def compute_naivebayes_model(self):
        """ Gaussian Naive Bayes Model """
        clf = GaussianNB() 
        self.compute_model(clf, clfid="GaussNB")

    def compute_knn_model(self):
        """ Implements K nearest Neighborhood Classifier """
        param_grid = {"n_neighbors":[5, 10, 20, 30, 50]}
        clf = KNeighborsClassifier()
        self.compute_model_pipeline(clf, param_grid, clfid="KNN")





if __name__ == "__main__":
    L = LearningFromData()
    L.prepare_service()
    L.get_data()
    L.compute_logistic_model()
    L.compute_naivebayes_model()
    L.compute_randomforest_model()
    L.compute_knn_model()
    plt.legend(loc="lower right")
    plt.savefig("figures/telemkting_roc.png", fmt="png")
    plt.show()

