from MLFE import load
from MLFE import Model
from MLFE import Buffer
from MLFE import Env
from MLFE import updateTargetGraph
from MLFE import updateTarget
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.linear_model import LogisticRegression,Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from cleanlab.pruning import get_noise_indices
from sklearn.metrics import matthews_corrcoef
from utils import *
from args import args
import tensorflow as tf
import os
import numpy as np
from sklearn import metrics
import tqdm
import copy
import warnings
import math

warnings.filterwarnings('ignore')
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"#args.cuda
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

tf_config = tf.ConfigProto()
tf_config.gpu_options.per_process_gpu_memory_fraction = 0.1 # 分配10%

'''
    单智能体
    CAFEM网络
'''


def main(ii):
    opt_type= 'o1'
    opt_size = 9 if opt_type =='o1' else 5
    qsa_size = 100
    input_size = 400
    buffer_size = 2000
    seed = 3

    file_path = 'seed.txt'          # 随机种子
    file_obj = open(file_path, 'a')
    file_obj.writelines(str(seed))
    file_obj.write("\n")
    file_obj.close()

    num_epochs = 50000
    n_jobs = 1
    tau = 0.05
    gamma = 0.9
    epsilon = 1
    batch_size = 100

    save_model = True
    train = True
    test = True
    out_dir = 'D:/FEL实验/1226/CAFEM-master1/src/out/ant-1-4/safem'+str(ii)
    model_dir = 'D:/FEL实验/1226/CAFEM-master1/src/out/ant-1-4/safem_model'+str(ii)
    if not os.path.isdir('D:/FEL实验/1226/CAFEM-master1/src/out/ant-1-4'):
        os.mkdir('D:/FEL实验/1226/CAFEM-master1/src/out/ant-1-4')
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    did = 1480
    f_dataset = "D:/FEL实验/1226/CAFEM-master1/promise1/ant-1.4-train"+str(ii)+".arff"
    dataset, tasktype = load(f_path=f_dataset)
    f_dataset1 = "D:/FEL实验/1226/CAFEM-master1/promise1/ant-1.4-test"+str(ii)+".arff"
    dataset_test, tasktype = load(f_path=f_dataset1)


    globalbuff = Buffer(buffer_size)
    if train:


        modelNetwork = Model(opt_size=opt_size, input_size=input_size, name="model", maml=False)
        targetNetwork = Model(opt_size=opt_size, input_size=input_size, name="target", maml=False)

        perf = 0
        pretransform = []
        with tf.Session(config=tf_config) as sess:
            saver = tf.train.Saver()

            saver.restore(sess, model_dir+ "/model.ckpt")



            pretransform_test = []

            for fid in tqdm.tqdm(range(dataset.shape[1] - 1), total=dataset.shape[1] - 1):
                env_test = Env(dataset, feature=fid,  opt_type=opt_type,tasktype=tasktype,
                               random_state=seed, pretransform=pretransform_test, n_jobs=n_jobs,
                               evaluatertype='rf')

                s = np.copy(env_test.state)
                act_mask = np.copy(env_test.action_mask)
                Q = sess.run(modelNetwork.Q_, feed_dict={modelNetwork.inputs: [s]})
                action = np.ma.masked_array(Q, mask=act_mask).argmax()
                s_next, reward = env_test.step(action)

                pretransform_test.append((fid, '_'.join(env_test.best_seq)))

            f = open(os.path.join(out_dir, "test_succeed_feat.csv"), 'a')
            for val in pretransform_test:
                f.write("%d,%s\n" % (val[0], val[1]))
            f.close()

            dataset1 = np.vstack((dataset, dataset_test))
            env1 = Env(dataset1, feature=0, opt_type=opt_type,tasktype=tasktype,
                       random_state=seed, pretransform=pretransform_test, n_jobs=n_jobs, evaluatertype='rf')
            dataset1_ = copy.deepcopy(env1.origin_dataset)
            print('dataset1_:', env1.origin_dataset.shape[1] - 1)
            kk = dataset.shape[0]
            X_train1, X_test1 = dataset1_[0:kk, 0:-1], dataset1_[kk:, 0:-1]
            y_train1, y_test1 = dataset1_[0:kk, -1], dataset1_[kk:, -1]
            rf1 =LogisticRegression(solver='liblinear',random_state = seed)
            rf1.fit(X_train1, y_train1)
            pre = rf1.predict(X_test1)
            final_pfm = metrics.f1_score(y_test1, pre, pos_label=1, average="binary")
            f = open(os.path.join(out_dir, "test_succeed.csv"), 'a')
            f.write("%d,%.6f\n" % (ii, final_pfm))
            f.close()

            mcc = matthews_corrcoef(y_test1, pre)
            f = open(os.path.join(out_dir, "test_mcc.csv"), 'a')
            f.write("%d,%.6f\n" % (ii, mcc))
            f.close()

            prob = rf1.predict_proba(X_test1)
            thresholds = metrics.roc_auc_score(y_test1, prob[:, -1])
            f = open(os.path.join(out_dir, "test_auc.csv"), 'a')
            f.write("%d,%.6f\n" % (ii, thresholds))
            f.close()


            for act in pretransform_test:
                print(act)


        tf.reset_default_graph()


if __name__ == "__main__":
    for ii in range(5):
        main(ii)