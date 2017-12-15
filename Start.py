import scipy.io as sio
import INRF
from sklearn.model_selection import ParameterGrid

data_dict=sio.loadmat('data/Imbalanced_data.mat')
data_all=data_dict['Imbalanced_data']
name_list=data_all[:,0]
IR_ratio=data_all[:,1]
data_set=data_all[:,2]
params = {
    'L': [5], # upper bound of level(auto)
    'N': [5], # bagging iterations
    'rp': [0.85,0.9,0.95,1,1.05,1.1,1.15], #disturbance eta（1不变,<1偏向多数类，>1偏向少数类）
}
params = list(ParameterGrid(params))
for name_index in range(3,len(name_list)):
    data_name = name_list[name_index][0]
    data = data_set[name_index]
    train_data_all=data[:,0]
    test_data_all=data[:,1]
    # combine the file name
    file_name = 'result/'+data_name+'.txt'
    INRF.main(data_name, file_name, train_data_all, test_data_all, params)