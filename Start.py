import scipy.io as sio
import INRF
import pandas as pd
from sklearn.model_selection import ParameterGrid

data_dict=sio.loadmat('data/Imbalanced_data.mat')
data_all=data_dict['Imbalanced_data']
name_list=data_all[:,0]
IR_ratio=data_all[:,1]
data_set=data_all[:,2]
base_classifier='RF'#base classifiers which can be used in our method. selected from 'RF','LR','NB'
if base_classifier=='NB' or 'LR':
    params={
        'L': [5],  # upper bound of level(auto)
        'N': [5],  # sample bagging iterations
        'rp': [0.85, 0.9, 0.95, 1, 1.05, 1.1, 1.15],  # disturbance eta（1不变,<1偏向多数类，>1偏向少数类)
        'F': [5], # feature bagging iterations (only used in Naive Bayes (NB) and Logistic Regression (LR) which work as base classifiers)
        'rf': [0.5, 0.7, 0.9],  # feature bagging rate (only used with NB and LR)
    }
else:
    params = {
        'L': [5],  # upper bound of level(auto)
        'N': [5],  # sample bagging iterations
        'rp': [0.85, 0.9, 0.95, 1, 1.05, 1.1, 1.15],  # disturbance eta（1不变,<1偏向多数类，>1偏向少数类)
    }
params = list(ParameterGrid(params))
datasets=['ecoli_0_vs_1','glass1','wisconsin','pima','glass0','vehicle3','ecoli1','new_thyroid2','yeast3','ecoli3','page_blocks0','yeast-0-2-5-6_vs_3-7-8-9',
         'ecoli-0-3-4-7_vs_5-6','glass-0-4_vs_5','yeast_0_5_6_7_9_vs_4','glass-0-6_vs_5','cleveland-0_vs_4','ecoli-0-1-4-6_vs_5','yeast_1_vs_7','ecoli4',
         'page_blocks_1_3_vs_4','glass_0_1_6_vs_5','yeast_2_vs_8','glass5','shuttle_c2_vs_c4','yeast4','yeast_1_2_8_9_vs_7','yeast5','ecoli_0_1_3_7_vs_2_6',
         'abalone19']
data_name_list=[]
best_macc_list=[]
best_macc_std_list=[]
for name_index in range(0,len(name_list)):
    data_name = name_list[name_index][0]
    if data_name not in datasets:continue
    data = data_set[name_index]
    train_data_all=data[:,0]
    test_data_all=data[:,1]
    # combine the file name
    file_name = 'result/'+data_name+'.txt'
    best_macc, best_macc_std=INRF.main(data_name, file_name, train_data_all, test_data_all, params, base_classifier=base_classifier)
    data_name_list.append(data_name)
    best_macc_list.append(best_macc)
    best_macc_std_list.append(best_macc_std)
result=pd.DataFrame([data_name_list,best_macc_list,best_macc_std_list],index=['data_name','best_macc','std'])
result=result.transpose()
result.to_csv('report/result_'+base_classifier+'.csv')