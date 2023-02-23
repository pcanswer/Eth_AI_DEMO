import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,precision_recall_fscore_support
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
import joblib
import datetime

dir = './udp_data/'
# 处理CICFlowMeter原始生成的数据，输入类型标签
def csv_process(label):
    df_normal = pd.read_csv(dir+'UDP_'+label+'.csv')
    df_normal
    df_normal.keys()
    df = df_normal.iloc[:,7:]
    df['Label'] = label
    X_train, X_test= train_test_split(df, train_size = 0.8, test_size = 0.2, random_state = 0)
    X_train.to_csv(dir+'UDP_'+label+'_train.csv',index=0)
    X_test.to_csv(dir+'UDP_'+label+'_test.csv',index=0)
    
# 处理合成训练、测试数据集
def data_concat(label):
    df1 = pd.read_csv('./udp_data/UDP_Scan_'+label+'.csv')
    df2 = pd.read_csv('./udp_data/UDP_Normal_'+label+'.csv')
    df3 = pd.read_csv('./udp_data/UDP_Flood_'+label+'.csv')
    df = pd.concat([df1,df2,df3])
    df.to_csv('./udp_data/UDP_process_'+label+'.csv',index=0)
    

if __name__=='__main__':
    csv_process('Flood')
    csv_process('Scan')
    csv_process('Normal')
    data_concat('test')
    data_concat('train')
