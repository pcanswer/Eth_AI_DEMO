# 导入需要使用的包
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,precision_recall_fscore_support
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
import joblib
import datetime

train_flag = 0
test_flag = 1
show_cm = 0
test_file_dir ='./udp_data/UDP_process_test.csv'  
model_file = './udp_model/udp_dt.joblib'
train_file = './udp_data/UDP_process_train.csv'
scaler_file = './udp_model/scaler_file.joblib'
encoder_file = './udp_model/encoder_file.joblib'
class Eth_DT:
    def preprocessing(self,df):
        # Min-max normalization
        standard_scaler = StandardScaler()
        features_index = df.drop(['Label'], axis=1).dtypes[df.dtypes != 'object'].index
        df[features_index] = standard_scaler.fit_transform(df[features_index])
        joblib.dump(standard_scaler, scaler_file)
        # 空值填成0
        df = df.fillna(0)
        print(df[['Label']].value_counts())
        # 对标签进行编码
        labelencoder = LabelEncoder()
        df.iloc[:, -1] = labelencoder.fit_transform(df.iloc[:, -1])
        joblib.dump(labelencoder, encoder_file)
        # X,Y 数据集划分
        X = df.drop(['Label'],axis=1).values 
        Y = df.iloc[:, -1].values.reshape(-1,1)
        Y = np.ravel(Y)
        return X,Y
    # 载入模型函数
    def load(self,model):
        self.model = model
    # 训练函数
    # 输入数据，标签
    def trianing(self,X,Y):
        # 划分数据集
        # 8:2划分为训练集和验证集
        X_train, X_test, Y_train, Y_test = train_test_split(X,Y, train_size = 0.8, test_size = 0.2, random_state = 0,stratify = Y)
        # 统计训练数据和标签的大小
        print(X_train.shape)
        print(pd.Series(Y_train).value_counts())
        # SMOTE过采样
        X_resampled, Y_resampled = SMOTE().fit_resample(X_train, Y_train)
        X_train = X_resampled
        Y_train = Y_resampled
        print(X_train.shape)
        # 重新统计数值
        print(pd.Series(Y_train).value_counts())
        # Decision tree training and prediction
        dt = DecisionTreeClassifier(random_state = 0)
        dt.fit(X_train,Y_train) 
        # 保存模型
        joblib.dump(dt,model_file)
        dt_score=dt.score(X_test,Y_test)
        # 预测数值
        Y_predict=dt.predict(X_test)
        Y_true=Y_test
        # 绘制DT的表现图
        print('Accuracy of DT: '+ str(dt_score))
        precision,recall,fscore,none= precision_recall_fscore_support(Y_true, Y_predict, average='weighted') 
        print('Precision of DT: '+(str(precision)))
        print('Recall of DT: '+(str(recall)))
        print('F1-score of DT: '+(str(fscore)))
        print(classification_report(Y_true,Y_predict))
        # 计算混淆矩阵
        cm=confusion_matrix(Y_true,Y_predict)
        f,ax=plt.subplots(figsize=(5,5))
        # 绘制混淆矩阵热力图
        sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
        plt.xlabel("y_pred")
        plt.ylabel("y_true")
        plt.show()
    # 检测函数，输入数据
    # 返回检测结果、输入时间的第一个时间点、攻击标识位
    def detect(self,df,standard_scaler,labelencoder):
        first_time = datetime.datetime.now()
        df_raw = df
        features_index = df.dtypes[df.dtypes != 'object'].index
        df[features_index] = standard_scaler.transform(df[features_index]) 
        df.iloc[:, -1] = labelencoder.transform(df.iloc[:, -1])
        # X,Y 数据集划分
        X = df.drop(['Label'],axis=1).values 
        Y = df.iloc[:, -1].values.reshape(-1,1)
        Y = np.ravel(Y)
        X = pd.DataFrame(df,columns=features_index).values 
        Y = df.Label.values.reshape(-1,1)
        out = self.model.predict(X)
        result = pd.DataFrame(out)
        result.columns = ['status']
        result = pd.concat([df_raw, result], axis=1)
        if show_cm == 1:
            dt = joblib.load(model_file)
            dt_score=dt.score(X,Y)
            # 预测数值
            Y_predict=out
            Y_true=Y
            # 绘制DT的表现图
            print('Accuracy of DT: '+ str(dt_score))
            precision,recall,fscore,none= precision_recall_fscore_support(Y_true, Y_predict, average='weighted') 
            print('Precision of DT: '+(str(precision)))
            print('Recall of DT: '+(str(recall)))
            print('F1-score of DT: '+(str(fscore)))
            print(classification_report(Y_true,Y_predict))
            # 计算混淆矩阵
            cm=confusion_matrix(Y_true,Y_predict)
            f,ax=plt.subplots(figsize=(5,5))
            # 绘制混淆矩阵热力图
            sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
            plt.xlabel("y_pred")
            plt.ylabel("y_true")
            plt.show()
        # 有无攻击判定
        if not result[result['status'] != 0].empty:
            attack_detect = 1
        else:
            attack_detect = 0
            # 检测结果赋值检测标签
        result.loc[result['Label'] == 2, 'Label'] = 'Scan'
        result.loc[result['Label'] == 0, 'Label'] = 'Flood'
        result.loc[result['Label'] == 1, 'Label'] = 'Normal'
        result.loc[result['status'] == 2, 'status'] = 'Scan'
        result.loc[result['status'] == 0, 'status'] = 'Flood'
        result.loc[result['status'] == 1, 'status'] = 'Normal'
        return result, first_time,attack_detect
# 主函数
if __name__ == '__main__':
    if train_flag == 1:
        df = pd.read_csv(train_file)
        model = Eth_DT()
        X,Y = model.preprocessing(df)
        model.trianing(X,Y)
    if test_flag == 1:
        df = pd.read_csv(test_file_dir)
        model = Eth_DT()
        model1 = joblib.load(model_file)
        scaler = joblib.load(scaler_file)
        labelencoder = joblib.load(encoder_file)
        model.load(model1)
        result,first_time,attack_detect = model.detect(df,scaler,labelencoder)
        print(result)