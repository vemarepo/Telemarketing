from __future__ import division
import pandas
import numpy as np
import logging
import matplotlib.pyplot as plt
from sklearn.feature_extraction import DictVectorizer as DV
from sklearn.preprocessing import StandardScaler
import itertools
def getmarker():
    markers = ['--', '-.', '-o', '-*', '-x', '-v']
    for marker in markers:
        yield marker

markgen = getmarker()

def rmse( x_test , x_pred):
    return np.sqrt(np.mean( (x_test - x_pred)**2  ))

def plot_coeff(coeff, fnm, title_stg="Linear"):
    plt.figure()
    print np.log10(np.abs(coeff))
    plt.stem(np.arange(len(coeff)), (np.abs(coeff)))
    plt.xlabel("Feature number")
    plt.ylabel("Coefficient of Feature")
    plt.savefig(fnm)
    plt.title(title_stg)
    plt.grid()
    plt.show()

def plot_roc_curve(fpr, tpr, thresholds, label="", title_stg="", ax=None):
    if ax is None:
        f = plt.figure()
        ax = f.add_subplot(111)
    mrk = markgen.next() 
    print "marker", mrk
    ax.plot(fpr, tpr, mrk, label =label)
    ax.set_xlabel("False Prob Rate")
    ax.set_ylabel("True Prob Rate")
    plt.title(title_stg)

def getlogger():
    # set up logging to file - see previous section for more details
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename='tmp/preprocess.log',
                        filemode='w')
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)
    return logging


def encode_categ_features(categ_feat, quant_feat, train_data, test_data, response_varb="target"):

	""" Encoding Categorical Features via the Dict Vectorizer Method """
	categ_train_dict = train_data.fillna('NA')[categ_feat].T.to_dict().values()
	categ_test_dict = test_data.fillna('NA')[categ_feat].T.to_dict().values()
        # Fit and Transform
	vectorizer = DV( sparse = False )
	categ_train = vectorizer.fit_transform( categ_train_dict )
	categ_train_df = pandas.DataFrame(categ_train, index=train_data.index)
 	 
	# Only Transform
	categ_test = vectorizer.transform( categ_test_dict )
	categ_test_df = pandas.DataFrame(categ_test, index = test_data.index)

	scaler = StandardScaler()
	qft = quant_feat[quant_feat != response_varb]
	scld_train_data = pandas.DataFrame( scaler.fit_transform(train_data[qft].values), 
										index = train_data.index,
										columns  = qft)
	scld_train_data[response_varb] = train_data[response_varb]

	new_train_df = pandas.concat( [scld_train_data, categ_train_df], axis=1)
 
	scld_test_data = pandas.DataFrame( scaler.transform(test_data[qft].values),
										index = test_data.index,
										columns  = qft)

	new_test_df = pandas.concat( [scld_test_data, categ_test_df],  axis=1)
	return new_train_df, new_test_df, scaler

def compute_entropy(data, varb, resp):
    grouped = data[[varb,resp]].groupby([varb])
    M = len(data)
    net_ent = 0
    df2 = data[[varb,resp]].pivot_table(columns=[resp], rows=[varb], aggfunc=len)
    print "Variable", varb
    print df2.to_string()

    for name, group in grouped:
        N = len(group)
        p = group[resp].sum()/N
        q = 1-p
        logp = np.log2(p)
        logq = np.log2(q)
        if p == 0:
            logp = 0
        elif q == 0:
            logq = 0

        grp_ent = p*logp + q*logq
        #print p, logp, q, logq, grp_ent
        net_ent += -(N/M)*grp_ent 
    return net_ent


