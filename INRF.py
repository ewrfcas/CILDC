import numpy as np
import time as time_sys
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import DataToolBox

def test(INRFStruct, test_data):
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

def fun(l,n, p, nTree, train_data, train_label):
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
        [predict_label,_] = test(INRFStruct, train_data)
        [_,_,macc]=DataToolBox.get_macc(train_label,predict_label)
        m_acc.append(macc)
        if i >= 2 and m_acc[-1] <= m_acc[-1 - 1]:
            INRFStruct.pop()
            return INRFStruct

    return INRFStruct


def main(data_name, file_name, train_data_all, test_data_all, params):
    kflod = 5
    f = open(file_name, "w")
    best_macc=0;best_auc=0
    for param in params:
        f.write("Dataset-"+data_name+" L-"+str(param['L'])+" N-"+str(param['N'])+" rp-"+str(param['rp'])+"\n")
        print("Dataset-"+data_name+" L-"+str(param['L'])+" N-"+str(param['N'])+" rp-"+str(param['rp']))
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
            nTree = 50
            INRFStruct = fun(param['L'],param['N'],param['rp'], nTree, train_data, train_label)
            time.append(time_sys.time()-start_time)
            level.append(len(INRFStruct))
            # 测试
            [predict_label, values]=test(INRFStruct,test_data)
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
        if np.mean(auc)>best_auc:
            best_auc=np.mean(auc)
            best_auc_param=param
    f.write("The best mean-accuracy is: " + str(round(best_macc,4)) + ", the best param is: " + str(best_macc_param)+ "\n")
    print("The best mean-accuracy is: " + str(round(best_macc,4)) + ", the best param is: " + str(best_macc_param))
    f.write("The best auc is: " + str(round(best_auc,4)) + ", the best param is: " + str(best_auc_param) + "\n")
    print("The best auc is: " + str(round(best_auc,4)) + ", the best param is: " + str(best_auc_param))
    f.close()