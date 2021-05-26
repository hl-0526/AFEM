import pandas as pd
import tensorflow as tf
import numpy as np
import random
import os
import math
from multiprocessing import Pool, cpu_count, Process
import multiprocessing

from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.linear_model import LogisticRegression,Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, f1_score, log_loss, roc_auc_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import cross_val_score
from scipy.io.arff import loadarff
from scipy import stats
import numpy.ma as ma
from sklearn import metrics
from sklearn.metrics import make_scorer
from sklearn.naive_bayes import GaussianNB
import copy

import os
from args import args
from utils import normalize

from utils import *
def one_mse_func():
    def one_relative_abs(y_true,y_pred):
        mae = mean_absolute_error(y_true,y_pred)
        one_mae = 1 - mae/np.mean(np.abs(y_true - np.mean(y_true)))
        #print(one_mae,np.abs(one_mae))
        return np.abs(one_mae)
        
    scorefunc = make_scorer(one_relative_abs, greater_is_better=False)
    return scorefunc

class Evaluater(object):
    """docstring for Evaluater"""
    def __init__(self, cv=5, stratified=True, n_jobs=1, tasktype="C", evaluatertype="rf", n_estimators=3,
                 random_state=np.random.randint(100000)):
        # tasktype = "C" or "R" for classification or regression
        # evaluatertype = 'rf', 'svm', 'lr' for random forest, SVM, logisticregression
        self.random_state = random_state
        self.cv = cv
        self.stratified = stratified
        self.n_jobs = n_jobs
        self.tasktype = tasktype
        if self.tasktype == "C":
            self.kf = StratifiedKFold(n_splits=self.cv, random_state=self.random_state, shuffle=True)
        else:
            self.kf = KFold(n_splits=self.cv, random_state=self.random_state, shuffle=True)

        if evaluatertype =="rf":
            if tasktype == "C":
                self.clf = RandomForestClassifier( n_estimators=9,random_state=self.random_state)
            elif tasktype == "R":
                self.clf = RandomForestRegressor(n_estimators=9,random_state=self.random_state)
        elif evaluatertype == "lr":
            if tasktype == "C":
                self.clf = LogisticRegression(solver='liblinear',random_state = self.random_state)
            elif tasktype =="R":
                self.clf = Lasso(random_state = self.random_state)
        elif evaluatertype == "nb":
            self.clf =  GaussianNB()
        #print(evaluatertype)
    # @profile

    def CV2(self, X, y):
        res = []
        feature_importance = []

        # Parallel(n_jobs=1)(delayed()() )
        '''
        for train_index, test_index in self.kf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            self.clf.fit(X_train, y_train)
            y_test_hat = self.clf.predict(X_test)
            #feature_importance.append(self.clf.feature_importances_)
            res.append(metrics.f1_score(y_test, y_test_hat, pos_label=1, average="binary"))
        '''

        self.clf.fit(X, y)
        y_test_hat = self.clf.predict(X)
        # feature_importance.append(self.clf.feature_importances_)
        res.append(metrics.f1_score(y, y_test_hat, pos_label=1, average="binary"))

        #self.feature_importance = np.array(feature_importance).mean(axis=0)
        return np.array(res).mean(axis=0)

def load(f_path):
    le = LabelEncoder()
    tasktype=''
    if f_path[-4:] =='arff':
        dataset = loadarff(f_path)
        df = pd.DataFrame(dataset[0])
        sample = df.values[:, :]
        tasktype = "C"
    m = sample.shape[0]
    k=0
    for i in range(m):
        sample[i][-1] = int(sample[i][-1])
        if sample[i][-1] == -1:
            sample[i][-1] = 0
            k=k+1
    sample = np.array(sample)
    sample = sample.astype(float)
    print(k)
    return sample, tasktype

# Bit flipping environment
class Env():
    def __init__(self, dataset,feature,binsize=10,opt_type='o1',tasktype="C",evaluatertype="rf",\
                 random_state=np.random.randint(100000),pretransform=None,n_jobs=1):
        if  opt_type=='o2':
            maxdepth = 1
        #print(feature)
        self.opt_type = opt_type
        self.action= ['fs','square','tanh','round','log','sqrt','mmn','sigmoid','zscore'] \
            if opt_type == 'o1' else ['fs','sum','diff','product','divide']
        self.action_size = len(self.action)
        self.tasktype=tasktype
        self.evaluatertype = evaluatertype
        self.random_state = random_state

        self.origin_dataset = copy.deepcopy(dataset)
        self.origin_feat = feature

        self._pretrf_mapper = [i for i in range(self.origin_dataset.shape[1]-1)]            #修改
        if pretransform is not None:
            for act in pretransform:
                feat_id = act[0]
                actions = act[1].split("_")
                self.fe(actions,feat_id)


        self.evaluater = Evaluater(random_state=random_state,tasktype=tasktype,evaluatertype=evaluatertype,n_jobs=n_jobs)


        self._init_pfm =self.evaluater.CV2(self.origin_dataset[:,:-1],self.origin_dataset[:,-1])
        self.init_pfm = self._init_pfm
        self.y = np.copy(self.origin_dataset[:,-1])
        self.binsize = binsize

        self.dataset = copy.deepcopy(self.origin_dataset)
        self.current_f = self.origin_feat
        qsa_rep = self._QSA()
        self.state = np.concatenate([qsa_rep],axis=None)
        self.action_mask = [0] * (len(self.action))          #修改
        self.best_seq = []


    def step(self, action):

        operator = self.action[action]

        if operator == 'fs':

            reward = 0

        else:
            newfeature = feature = copy.deepcopy(self.dataset[:,self.current_f])

            if operator in set(['square', 'tanh', 'round']):
                newfeature = getattr(np, operator)(feature)
            elif operator == "log":
                vmin = feature.min()
                newfeature = np.log(feature - vmin + 1) if vmin < 1 else np.log(feature)

            elif operator == "sqrt":
                vmin = feature.min()
                newfeature = np.sqrt(feature - vmin) if vmin <0 else np.sqrt(feature)

            elif operator == "mmn":                                   #修改
                if feature.max() != feature.min():
                    mmn = MinMaxScaler()
                    newfeature = mmn.fit_transform(feature[:, np.newaxis]).flatten()
                else:
                    reward = -10000
                    return self.state, reward

            elif operator == "sigmoid":
                newfeature = (1 + getattr(np, 'tanh')(feature / 2)) / 2

            elif operator == 'zscore':                      #修改
                if np.var(feature) != 0:

                    newfeature = stats.zscore(feature)
                else:
                    reward =-10000
                    return  self.state,reward


            if newfeature is not None:

                newfeature = np.nan_to_num(newfeature)
                newfeature = np.clip(newfeature, -math.sqrt(3.4e38), math.sqrt(3.4e38))
                self.dataset = np.delete(self.dataset,self.current_f, axis=1)
                self.dataset = np.insert(self.dataset,self.current_f,newfeature,axis=1)

            else: #TODO
                pass


            X =copy.deepcopy(self.dataset[:, :-1])

            performance = self.evaluater.CV2(X, self.y)

            reward = (performance - self.init_pfm)*100




        qsa_rep = self._QSA()

        self.state = np.concatenate([qsa_rep],axis=None)

        self.best_seq = []
        self.best_seq.insert(0,operator)

        return self.state,reward

    def _QSA(self):
        QSA=[]
        for i in range(self.dataset.shape[1]-1):
            newfeats=copy.deepcopy(self.dataset[:, i])

            feat_0 = newfeats[self.y == 0]
            feat_1 = newfeats[self.y == 1]


            minval,maxval = feat_0.min(),feat_0.max()
            if abs(maxval - minval) < 1e-8:
                QSA0 = [0] * self.binsize
            else:
                bins = np.arange(minval,maxval,(maxval-minval) * 1.0 / self.binsize)[1:self.binsize]
                QSA0 = np.bincount(np.digitize(feat_0,bins)).astype(float) / len(feat_0)

            minval,maxval = feat_1.min(),feat_1.max()
            if abs(maxval - minval) < 1e-8:
                QSA1 = [0] * self.binsize
            else:
                bins = np.arange(minval,maxval,(maxval-minval) * 1.0 / self.binsize)[1:self.binsize]
                QSA1 = np.bincount(np.digitize(feat_1,bins)).astype(float) / len(feat_1)
            QSA = np.concatenate([QSA,QSA0])
            QSA = np.concatenate([QSA, QSA1])
        return QSA


    def fe(self,operators,feat_id):
        if  type(feat_id) is int:
            new_feat_id = self._pretrf_mapper[feat_id]
            feature = copy.deepcopy(self.origin_dataset[:, new_feat_id])


        for operator in operators:
            #print(operator)
            if type(feat_id) is int:
                if operator in set(['square', 'tanh', 'round']):
                    feature = getattr(np, operator)(feature)
                elif operator == "log":
                    vmin = feature.min()
                    feature = np.log(feature - vmin + 1) if vmin < 1 else np.log(feature)

                elif operator == "sqrt":
                    vmin = feature.min()
                    feature = np.sqrt(feature - vmin) if vmin <0 else np.sqrt(feature)

                elif operator == "mmn":
                    if feature.max() != feature.min():
                        mmn = MinMaxScaler()
                        feature = mmn.fit_transform(feature[:, np.newaxis]).flatten()
                    else:
                        feature = None

                elif operator == "sigmoid":
                    feature = (1 + getattr(np, 'tanh')(feature / 2)) / 2

                elif operator == 'zscore':
                    if np.var(feature) != 0:
                        feature = stats.zscore(feature)
                    else:
                        feature = None
                else:
                    feature =None
        if len(operators) > 0 and feature is not None and operators[0] != 'fs':
            feature = np.nan_to_num(feature)
            feature = np.clip(feature, -math.sqrt(3.4e38), math.sqrt(3.4e38))
            self.origin_dataset = np.delete(self.origin_dataset, feat_id, axis=1)
            self.origin_dataset = np.insert(self.origin_dataset,feat_id,feature,axis=1)

        return feature




# Experience replay buffer
class Buffer():
    def __init__(self, buffer_size = 50000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        self.buffer.append(experience)
        if len(self.buffer) > self.buffer_size:
            self.buffer = self.buffer[1:]
        #print(len(self.buffer))

    def sample(self,size):
        if len(self.buffer) >= size:
            experience_buffer = self.buffer
        else:
            experience_buffer = self.buffer * size
        return np.copy(np.reshape(np.array(random.sample(experience_buffer,size)),[size,5]))

# Simple feed forward neural network

class Model():
    def __init__(self, opt_size, input_size, name, meta=False,update_lr=1e-3,meta_lr=0.001,num_updates=1,maml=True,qsasize=280):
        self.input_size = input_size
        self.opt_size = self.dim_output =  opt_size
        self.dim_hidden = [256]
        self.skip=1
        self.qsasize = qsasize
        self.inputs = tf.placeholder(shape=[None, self.input_size], dtype=tf.float32)
        self.Q_next = tf.placeholder(shape=None, dtype=tf.float32)
        self.action = tf.placeholder(shape=None, dtype=tf.int32)
        self.inputsa = tf.placeholder(shape=[None,None, self.input_size], dtype=tf.float32)
        self.inputsb = tf.placeholder(shape=[None,None, self.input_size], dtype=tf.float32)
        self.Q_nexta = tf.placeholder(shape=[None,None], dtype=tf.float32)
        self.Q_nextb = tf.placeholder(shape=[None,None], dtype=tf.float32)
        self.actiona = tf.placeholder(shape=[None,None], dtype=tf.int32)
        self.actionb = tf.placeholder(shape=[None,None], dtype=tf.int32)
        self.update_lr = update_lr
        self.meta_lr = meta_lr
        self.num_updates = num_updates
        self.size = opt_size
        self.input_size = input_size
        self.loss_func = self.mse
        self.weights = self.construct_fc_weights()
        self.network()
        if maml:
            self.construct_model()
        self.init_op = tf.global_variables_initializer()

    def mse(self, y_pred, y_true):
        return tf.reduce_sum(tf.square(y_pred - y_true))

    def construct_fc_weights(self):
        factor = 1
        weights = {}
        weights['w1'] = tf.Variable(tf.truncated_normal([self.input_size, self.dim_hidden[0]], \
                                                        stddev=math.sqrt(  factor/((self.input_size+self.dim_hidden[0])/2)  )))
        weights['b1'] = tf.Variable(tf.zeros([self.dim_hidden[0]]))
        for i in range(1, len(self.dim_hidden)):
            weights['w' + str(i + 1)] = tf.Variable(
                tf.truncated_normal([self.dim_hidden[i - 1], self.dim_hidden[i]], \
                                    stddev=math.sqrt( factor/((self.dim_hidden[i - 1]+ self.dim_hidden[i])/2 ) )))
            weights['b' + str(i + 1)] = tf.Variable(tf.zeros([self.dim_hidden[i]]))
        weights['w' + str(len(self.dim_hidden) + 1)] = tf.Variable(
            tf.truncated_normal([self.dim_hidden[-1], self.dim_output], \
                                stddev=math.sqrt( factor/((self.dim_hidden[-1]+ self.dim_output)/2 ) )))
        #weights['b' + str(len(self.dim_hidden) + 1)] = tf.Variable(tf.zeros([self.dim_output]))

        return weights

    def forward(self, inp,  weights, reuse=False):
        hidden = normalize(tf.matmul(inp, weights['w1']) + weights['b1'], activation=tf.nn.relu, reuse=reuse,
                           scope='0',norm="None")

        for i in range(1, len(self.dim_hidden)):
            hidden = normalize(tf.matmul(hidden, weights['w' + str(i + 1)]) + weights['b' + str(i + 1)],
                               activation=tf.nn.relu, reuse=reuse, scope=str(i + 1), norm='None')

        Q_ = tf.matmul(hidden, weights['w' + str(len(self.dim_hidden) + 1)])

        return Q_
    def L2loss(self,weights,reg):
         loss_reg = 0.0
         for key in weights:
             loss_reg += reg*tf.reduce_sum(tf.square(weights[key]))
         return loss_reg

    def network(self):

        self.Q_ = self.forward(self.inputs, self.weights)
        self.action_onehot = tf.one_hot(self.action, self.size, dtype=tf.float32)
        self.Q = tf.reduce_sum(tf.multiply(self.Q_, self.action_onehot), axis=1)
        self.loss = self.loss_func(self.Q_next, self.Q) +self.L2loss(self.weights,1e-5)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.meta_lr)
        self.train_op = self.optimizer.minimize(self.loss)
        #TODO optimizer can be only one






def updateTargetGraph(tfVars, tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx, var in enumerate(tfVars[0:total_vars // 2]):
        op_holder.append(tfVars[idx + total_vars // 2].assign(
            (var.value() * (1. - tau)) + (tau * tfVars[idx + total_vars // 2].value())))
    return op_holder

def updateTarget(op_holder, sess):
    for op in op_holder:
        sess.run(op)
