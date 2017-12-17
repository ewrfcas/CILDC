import numpy as np
import time as time_sys
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
import DataToolBox

def test_RF(INRFStruct, test_data):
    L=len(INRFStruct)
    N=len(INRFStruct[0])
    test_data_temp = test_data
    for i in range(0,L):
        values = np.zeros((test_data.shape[0], 2))
        for j in range(0,N):
            INRFStruct_temp=INRFStruct[i][j]
            values_temp = INRFStruct_temp.predict_proba(test_data_temp)
            values = values + values_temp
        values = values / N
        test_data_temp = np.hstack((test_data, values))
    predict_label=np.empty([values.shape[0]])
    for i in range(0,values.shape[0]):
        if values[i, 1] > values[i, 0]:
            predict_label[i] = 1
        else:
            predict_label[i] = 0
    return predict_label,values

def fun_RF(l,n, p, nTree, train_data, train_label):
    train_data_alter = train_data
    m_acc = []
    INRFStruct=[]
    for i in range(0,l):
        if i!=l-1:
            # 插值特征(置信度)
            inter_feature = np.zeros((len(train_label),2))
            # 内部交叉验证
            skf = StratifiedKFold(shuffle=True,n_splits=3,random_state=11810)
            for train_index, test_index in skf.split(X=train_data_alter, y=train_label):
                data_train_temp = train_data_alter[train_index,:]
                label_train_temp = train_label[train_index]
                data_test_temp = train_data_alter[test_index,:]
                # 区分多数类少数类，返回[多数类，少数类，多数类类标，少数类类标]
                [data_much, data_less, much_label, less_label] = DataToolBox.divide_data(data_train_temp, label_train_temp)
                for j in range(0,n):
                    # 随机下采样
                    [tData, tLabel] = DataToolBox.RUS(data_much, data_less, much_label, less_label)
                    rf_temp=RandomForestClassifier(n_estimators=nTree)
                    rf_temp.fit(tData,np.ravel(tLabel))
                    inter_feature_temp = rf_temp.predict_proba(data_test_temp)
                    inter_feature_temp[:, int(much_label)] = inter_feature_temp[:, int(much_label)] * p
                    inter_feature_temp[:, int(less_label)] = inter_feature_temp[:, int(less_label)] / p
                    inter_feature[test_index,:]=inter_feature[test_index,:]+inter_feature_temp
        # 区分多数类少数类，返回[多数类，少数类，多数类类标，少数类类标]
        [data_much, data_less, much_label, less_label] = DataToolBox.divide_data(train_data_alter, train_label)
        INRFStruct_oneLine = []
        for j in range(0,n):
            # 随机下采样
            [tData, tLabel] = DataToolBox.RUS(data_much, data_less, much_label, less_label)
            INRFStruct_temp = RandomForestClassifier(n_estimators=nTree)
            INRFStruct_temp.fit(tData,np.ravel(tLabel))
            INRFStruct_oneLine.append(INRFStruct_temp)
        INRFStruct.append(INRFStruct_oneLine)
        # 插值
        inter_feature = inter_feature / n
        train_data_alter = np.hstack((train_data, inter_feature))
        # 训练集评估
        [predict_label,_] = test_RF(INRFStruct, train_data)
        [_,_,macc]=DataToolBox.get_macc(train_label,predict_label)
        m_acc.append(macc)
        if i >= 2 and m_acc[-1] <= m_acc[-1 - 1]:
            INRFStruct.pop()
            return INRFStruct

    return INRFStruct

def test(NBCCIStruct,feature,test_data):
    test_data_temp=test_data
    L=len(NBCCIStruct)#层数
    N=len(NBCCIStruct[0])#样本bagging次数
    for i in range(0,L):
        values = np.zeros((test_data.shape[0], 2))
        for j in range(0,N):
            values =values+ train_test(NBCCIStruct[i][j], feature[i][j], test_data_temp)
        values=values/N
        test_data_temp = np.hstack((test_data, values))
    predict_label = np.empty([values.shape[0]])
    for i in range(0, values.shape[0]):
        if values[i, 1] > values[i, 0]:
            predict_label[i] = 1
        else:
            predict_label[i] = 0
    return predict_label, values

#训练内部插值
def train_test(NBStruct,feature_temp,data):
    F=len(NBStruct)
    confidence=np.zeros((data.shape[0],2))
    for i in range(0,F):
        confidence=confidence+NBStruct[i].predict_proba(data[:,feature_temp[i]])
    confidence=confidence/F

    return confidence

def NB_FBagging(data,label,f,rf):
    fp=int(data.shape[1]*rf)
    NBStruct=[]
    feature=[]
    for i in range(0,f):
        #特征随机
        index = np.arange(data.shape[1])
        np.random.shuffle(index)
        data_temp=data[:,index[0:fp]]
        feature.append(index[0:fp])
        clf = GaussianNB()
        clf.fit(data_temp,np.ravel(label))
        NBStruct.append(clf)

    return feature,NBStruct

def LR_FBagging(data,label,f,rf):
    fp = int(data.shape[1] * rf)
    LRStruct = []
    feature = []
    for i in range(0, f):
        # 特征随机
        index = np.arange(data.shape[1])
        np.random.shuffle(index)
        data_temp = data[:, index[0:fp]]
        feature.append(index[0:fp])
        clf = LogisticRegression(C=10000)
        clf.fit(data_temp, np.ravel(label))
        LRStruct.append(clf)

    return feature, LRStruct

def fun(l,f,n,rf,rp,train_data, train_label,base_classifier):
    train_data_alter = train_data
    m_acc = []
    NBCCIStruct=[]
    feature=[]
    L=l
    for i in range(0,L):
        if i!=L-1:
            # 插值特征(置信度)
            inter_feature = np.zeros((len(train_label),2))
            # 内部交叉验证 3cv
            skf = StratifiedKFold(shuffle=True,n_splits=3)
            for train_index, test_index in skf.split(X=train_data_alter, y=train_label):
                data_train_temp = train_data_alter[train_index,:]
                label_train_temp = train_label[train_index]
                data_test_temp = train_data_alter[test_index,:]
                # 区分多数类少数类，返回[多数类，少数类，多数类类标，少数类类标]
                [data_much, data_less, much_label, less_label] = DataToolBox.divide_data(data_train_temp, label_train_temp)
                for j in range(0,n):
                    # 随机下采样
                    [tData, tLabel] = DataToolBox.RUS(data_much, data_less, much_label, less_label)
                    if base_classifier=='NB':
                        feature_temp, NBStruct_temp = NB_FBagging(tData, tLabel, f, rf)  # 特征bagging NB
                    elif base_classifier=='LR':
                        feature_temp, NBStruct_temp = LR_FBagging(tData, tLabel, f, rf)  # 特征bagging LR
                    inter_feature_temp = train_test(NBStruct_temp,feature_temp,data_test_temp)
                    inter_feature_temp[:, int(much_label)] = inter_feature_temp[:, int(much_label)] * rp
                    inter_feature_temp[:, int(less_label)] = inter_feature_temp[:, int(less_label)] / rp
                    inter_feature[test_index,:]=inter_feature[test_index,:]+inter_feature_temp
        # 区分多数类少数类，返回[多数类，少数类，多数类类标，少数类类标]
        [data_much, data_less, much_label, less_label] = DataToolBox.divide_data(train_data_alter, train_label)
        NBCCIStruct_oneLine = []
        feature_oneLine=[]
        for j in range(0,n):
            # 随机下采样
            [tData, tLabel] = DataToolBox.RUS(data_much, data_less, much_label, less_label)
            if base_classifier=='NB':
                feature_temp, NBStruct_temp = NB_FBagging(tData, tLabel, f, rf)  # 特征bagging NB
            elif base_classifier=='LR':
                feature_temp, NBStruct_temp = LR_FBagging(tData, tLabel, f, rf)  # 特征bagging NB
            NBCCIStruct_oneLine.append(NBStruct_temp)
            feature_oneLine.append(feature_temp)
        NBCCIStruct.append(NBCCIStruct_oneLine)
        feature.append(feature_oneLine)
        # 插值
        if L!=1:
            inter_feature = inter_feature / n
            train_data_alter = np.hstack((train_data, inter_feature))
        # 训练集评估
        [predict_label,_] = test(NBCCIStruct,feature,train_data)
        [_,_,macc]=DataToolBox.get_macc(train_label,predict_label)
        m_acc.append(macc)
        if i >= 2 and m_acc[-1] <= m_acc[-1 - 1]:
            NBCCIStruct.pop()
            feature.pop()
            return NBCCIStruct,feature

    return NBCCIStruct,feature


def main(data_name, file_name, train_data_all, test_data_all, params, base_classifier='RF'):
    kflod = 5
    f = open(file_name, "w")
    best_macc=0;best_auc=0;best_macc_std=0
    for param in params:
        f.write(base_classifier+':'+data_name+' '+str(param)+"\n")
        print(base_classifier+':'+data_name+' '+str(param))
        f.write("--------------------------------------\n") # 分隔符
        print("--------------------------------------")
        time = [];acc = [];tp = [];tn = [];level = [];m_acc = [];auc = []
        for i_iter in range(0,kflod):# 第i_iter轮
            train_data = train_data_all[i_iter][:,0:-1]
            train_label = train_data_all[i_iter][:,-1]
            test_data = test_data_all[i_iter][:,0:-1]
            test_label = test_data_all[i_iter][:,-1]
            # 训练开始 计时
            start_time=time_sys.time()
            # 集成
            if base_classifier=='RF':
                nTree = 50
                INRFStruct = fun_RF(param['L'],param['N'],param['rp'], nTree, train_data, train_label)
                # 测试
                [predict_label, values]=test_RF(INRFStruct,test_data)
            else:
                INRFStruct, feature = fun(param['L'],param['F'], param['N'], param['rf'], param['rp'], train_data, train_label, base_classifier)
                # 测试
                [predict_label, values] = test(INRFStruct, feature, test_data)
            time.append(time_sys.time() - start_time)
            level.append(len(INRFStruct))
            scores = values[:,1]
            acc.append(DataToolBox.get_acc(test_label,predict_label))
            auc.append(DataToolBox.get_auc(test_label,scores))
            [tp_temp,tn_temp,macc_temp]=DataToolBox.get_macc(test_label,predict_label)
            tp.append(tp_temp)
            tn.append(tn_temp)
            m_acc.append(macc_temp)
            f.write("acc:" + str(round(acc[i_iter],4)) + " tp:" + str(round(tp[i_iter],4)) + " tn:" + str(round(tn[i_iter],4)) + " m-acc:"
                    + str(round(m_acc[i_iter],4)) + " auc:" + str(round(auc[i_iter],4)) + " level:" + str(round(level[i_iter],4)) + "\n")
            print("acc:" + str(round(acc[i_iter],4)) + " tp:" + str(round(tp[i_iter],4)) + " tn:" + str(round(tn[i_iter],4)) + " m-acc:"
                    + str(round(m_acc[i_iter],4)) + " auc:" + str(round(auc[i_iter],4)) + " level:" + str(round(level[i_iter],4)))
        f.write("The average accuracy is: "+str(round(np.mean(acc),4))+"±"+str(round(np.std(acc),4))+"\n")
        print("The average accuracy is: "+str(round(np.mean(acc),4))+"±"+str(round(np.std(acc),4)))
        f.write("The average mean-accuracy is: " + str(round(np.mean(m_acc),4)) + "±" + str(round(np.std(m_acc),4)) + "\n")
        print("The average mean-accuracy is: " + str(round(np.mean(m_acc),4)) + "±" + str(round(np.std(m_acc),4)))
        f.write("The average auc is: " + str(round(np.mean(auc),4)) + "±" + str(round(np.std(auc),4)) + "\n")
        print("The average auc is: " + str(round(np.mean(auc),4)) + "±" + str(round(np.std(auc),4)))
        f.write("The average level is: " + str(round(np.mean(level),4)) + "±" + str(round(np.std(level),4)) + "\n")
        print("The average level is: " + str(round(np.mean(level),4)) + "±" + str(round(np.std(level),4)))
        f.write("The average time(s) is: " + str(round(np.mean(time),4)) + "\n")
        print("The average time(s) is: " + str(round(np.mean(time),4)))
        f.write("--------------------------------------\n") # 分隔符
        print("--------------------------------------")
        if np.mean(m_acc)>best_macc:
            best_macc=np.mean(m_acc)
            best_macc_param=param
            best_macc_std=np.std(m_acc)
        if np.mean(auc)>best_auc:
            best_auc=np.mean(auc)
            best_auc_param=param
    f.write("The best mean-accuracy is: " + str(round(best_macc,4)) + ", the best param is: " + str(best_macc_param)+ "\n")
    print("The best mean-accuracy is: " + str(round(best_macc,4)) + ", the best param is: " + str(best_macc_param))
    f.write("The best auc is: " + str(round(best_auc,4)) + ", the best param is: " + str(best_auc_param) + "\n")
    print("The best auc is: " + str(round(best_auc,4)) + ", the best param is: " + str(best_auc_param))
    f.close()
    return best_macc,best_macc_std