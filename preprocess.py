import numpy as np
import pandas
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import logging
from ConfigParser import SafeConfigParser
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA
from sklearn.feature_extraction import DictVectorizer as DV
from sklearn.preprocessing import scale, StandardScaler
from utils import getlogger, encode_categ_features, compute_entropy

CONFIG_FILENAME="input.cfg"
logging = getlogger()

class PreProcessor:
    """ The input data is preprocessed before handled by the classifier/regressor 
        (o) Handling Missing Data 
        (o) Handling Categorical Data
        (o) Scaling the Data 
    """

    def __init__(self, fillnan_method="mean"):
        self.fillnan_method  = fillnan_method
        self.configobj = None
        self.raw_train_data = None
        self.raw_test_data = None

    def parse_config(self, config):
        """ Read the config file, which contains locations of the dataset and other data features """
        self.configobj = SafeConfigParser()
        self.configobj.read(config)
        
    @staticmethod
    def getdata(train_fnm, test_fnm, resp_varb):
        train_data = pandas.read_csv(train_fnm,  delimiter=";", header=0)
        n = np.zeros(len(train_data))
        n[(train_data[resp_varb] == "yes").values]  = 1
        train_data[resp_varb] = n

        
        test_data = pandas.read_csv(test_fnm, delimiter=";", header=0)
        n = np.zeros(len(test_data))
        n[(test_data[resp_varb] == "yes").values]  = 1
        test_data[resp_varb] = n
        
        logging.info(train_data.head().to_string())
        return train_data, test_data

    def handle_missing_data(self):
        """ Filling the missing data with mean or dropping them """
        tr = self.raw_train_data
        tst = self.raw_test_data
        if self.fillnan_method  == "mean":
            self.raw_train_data = tr.fillna(tr.mean())
            self.raw_test_data = tst.fillna(tst.mean())
        elif self.fillnan_method  == "drop":
            self.raw_train_data.dropna()
            self.raw_test_data.dropna()

    def handle_categorical_data(self):
        """ Encoding the categorical data using sklearn dict vectorizer """
        def filter_response(varbs):
            return np.array([x for x in varbs if x != self.response_varb])

        w = (self.raw_train_data.dtypes == np.dtype('object'))
        self.categ_feat = filter_response(w.index[w].values)
        self.quant_feat = filter_response(w.index[w == False].values)
        
        logging.info("Categorical Variables %s" %(self.categ_feat))
        logging.info("Quantitative Variables %s" %(self.quant_feat))
        self.enc_train_data, self.enc_test_data, self.scaler = encode_categ_features(self.categ_feat, self.quant_feat, 
                                                self.raw_train_data, 
                                                self.raw_test_data,
                                                self.response_varb)
    def adding_quadratic_features(self):
        """ Including the ability to add quadratic features """
        quant_feat = self.quant_feat[self.quant_feat != self.response_varb] 
        for v in quant_feat:
            self.enc_train_data["%s_2"%(v)] = self.enc_train_data[v]**2
            self.enc_test_data["%s_2"%(v)] = self.enc_test_data[v]**2


    def split_dataset(self):
        """ Split the training data set for optimization of data set parameters """
        cnames = self.enc_train_data.columns.values
        self.feature_names = cnames[cnames != self.response_varb]
        x = self.enc_train_data[self.feature_names].values
        y = self.enc_train_data[self.response_varb].values
        self.train_data, self.test_data, self.train_target, self.test_target = train_test_split(x, y, 
                                test_size=0.3, random_state=10)
        #self.scaler = StandardScaler()
        #self.train_data = self.scaler.fit_transform(self.train_data)
        #self.test_data = self.scaler.transform(self.test_data)
        
        #return self.train_data, self.test_data, self.train_target, self.test_target


    def process_data(self):
        """ All the processing and cleaning of data """
        self.parse_config(CONFIG_FILENAME)      
        train_fnm = self.configobj.get('data', 'train_filename')    
        test_fnm = self.configobj.get('data', 'test_filename')    
        self.response_varb = self.configobj.get('data', 'response_varb')
        #self.ignore_varbs = self.configobj.get('data', 'ignore_varbs')
        self.raw_train_data, self.raw_test_data = self.getdata(train_fnm, test_fnm, self.response_varb)
        self.handle_missing_data()
        self.handle_categorical_data()
        self.split_dataset()
        print self.raw_train_data.columns
        return self.train_data, self.test_data, self.train_target, self.test_target, self.enc_test_data.values

    def check_pca_dimensionality(self):
        pca =  PCA(whiten=True)
        scaled = scale(self.train_data, axis=0, with_mean=True, with_std=True)
        pca.fit(scaled)
        plt.figure(1, figsize=(4, 3))
        plt.clf()
        plt.axes([.2, .2, .7, .7])
        print pca.explained_variance_.cumsum()
        exp_var = pca.explained_variance_/pca.explained_variance_.sum() 
        plt.plot(exp_var.cumsum(), linewidth=2)
        plt.axis('tight')
        plt.xlabel('n_components')
        plt.ylabel('explained_variance_')
        plt.grid()
        plt.show()

    def plot_variances(self):
        var_tr = np.var(self.train_data, axis=0)
        var_tst = np.var(self.test_data, axis=0)
        plt.figure()
        plt.subplot(3, 1, 1)
        plt.plot(var_tr, "." )
        plt.ylabel('Variance Train')
        plt.subplot(3, 1, 2)
        plt.plot(var_tst, "." )
        plt.ylabel('Variance Test')
        plt.subplot(3, 1, 3)
        plt.plot(var_tst/var_tr, "." )
        plt.ylabel('Ratio of Variances')
        plt.show()





    def plot_categdata_analysis(self):
        #x = self.raw_train_data["job"]
        #y = self.raw_train_data["y"]
        #print len(x), len(y)
        #n_categ = len(self.categ_feat)
        #f, axs= plt.subplots(n_categ, 1, figsize=(8, 6))
        #sns.barplot(x, y, ci=None, palette="Paired", ax=axs[0])
        entrpy = [] 
        for varb in self.categ_feat:
            varb_entr = compute_entropy(self.raw_train_data, varb, self.response_varb) 
            entrpy.append(varb_entr)

        print entrpy, len(entrpy)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        S = pandas.Series(entrpy, index=self.categ_feat)
        print S
        S.plot(kind="bar", ax=ax)
        ax.set_xticklabels(self.categ_feat, rotation=45)
        plt.title("Entropy of Categorical Variables")
        plt.ylabel("Entropy")
        plt.show() 

    def plot_quantvarb_analysis(self):
        data  = self.raw_train_data
        resp = self.response_varb
        N = len(self.quant_feat)
        print data.columns
        for i, varb in enumerate(self.quant_feat):
            print varb
            grouped = data[[varb,resp]]
            gp0 = grouped[grouped[resp] == 0]
            gp1 = grouped[grouped[resp] == 1]
            #ax = axs[i]
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.hist(gp0[varb].values, bins=10, normed=True, color="#6495ED", alpha=.5, label="y==0")
            ax.hist(gp1[varb].values, bins=10, normed=True, color="#F08080", alpha=.5, label="y==1")
            plt.title("Variable %s"%(varb))
            ax.legend(loc='upper right')
            
            plt.savefig("figures/quant_varb_%s.jpg"%(varb), fmt="jpg")

        plt.show()
            






    def run(self):
        self.process_data()
        #self.plot_quantvarb_analysis()
        self.plot_categdata_analysis() 
        return 



if __name__ == "__main__":
    C = PreProcessor()
    C.run()
    #C.check_pca_dimensionality()
    #C.plot_variances()

