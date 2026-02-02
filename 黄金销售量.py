#!/usr/bin/env python
# coding: utf-8

# #  数据展示

# In[193]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
import calendar
# 设置中文显示和图形样式
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style('whitegrid')

# 加载数据
df = pd.read_excel('data.xlsx', sheet_name='Sheet1')

# 数据清洗
df['交易日期'] = pd.to_datetime(df['交易日期'])
df = df.groupby('交易日期').mean().reset_index()  # 处理重复日期
df = df.sort_values('交易日期').reset_index(drop=True)

# 添加时间特征
df['月份'] = df['交易日期'].dt.month
df['季度'] = df['交易日期'].dt.quarter
df['星期'] = df['交易日期'].dt.dayofweek + 1  # 1-7表示周一到周日
df['是否周末'] = df['星期'].isin([6, 7]).astype(int)
df['月日'] = df['交易日期'].dt.strftime('%m-%d')

# 检查数据
print("数据概览:")
print(df.head())
print("\n数据信息:")
print(df.info())
print("\n基本统计描述:")
print(df.describe())


# In[195]:


plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
plt.figure(figsize=(15, 8))
plt.plot(df['交易日期'], df['销售量（克）'], marker='o', linestyle='-', linewidth=1, markersize=4)
plt.title('销售量时间序列趋势', fontsize=15)
plt.xlabel('日期', fontsize=12)
plt.ylabel('销售量（克）', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# In[196]:


plt.figure(figsize=(15, 10))

# 直方图
plt.subplot(2, 2, 1)
sns.histplot(df['销售量（克）'], bins=20, kde=True)
plt.title('销售量分布直方图')

# 箱线图
plt.subplot(2, 2, 2)
sns.boxplot(y=df['销售量（克）'])
plt.title('销售量箱线图')

# 密度图
plt.subplot(2, 2, 3)
sns.kdeplot(df['销售量（克）'], shade=True)
plt.title('销售量密度图')

# QQ图
plt.subplot(2, 2, 4)
import scipy.stats as stats
stats.probplot(df['销售量（克）'], dist="norm", plot=plt)
plt.title('QQ图 - 正态性检验')

plt.tight_layout()
plt.show()


# # 缺失值处理

# In[199]:


import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.impute import KNNImputer
from statsmodels.tsa.seasonal import seasonal_decompose

# 重新加载原始数据
df = pd.read_excel('data.xlsx', sheet_name='Sheet1')
df['交易日期'] = pd.to_datetime(df['交易日期'])
df = df.groupby('交易日期').mean().reset_index()  # 处理重复日期
df = df.sort_values('交易日期').reset_index(drop=True)

# 检查日期连续性
print("原始数据日期范围:", df['交易日期'].min(), "至", df['交易日期'].max())
print("原始数据天数:", len(df))
print("理论工作日天数:", len(pd.date_range(start=df['交易日期'].min(), end=df['交易日期'].max(), freq='B')))


# In[201]:


# 创建完整的工作日日期范围
full_dates = pd.date_range(start=df['交易日期'].min(), end=df['交易日期'].max(), freq='B')
missing_dates = full_dates.difference(df['交易日期'])
print("\n缺失日期数量:", len(missing_dates))
print("缺失日期示例:", missing_dates[:5])  # 显示前5个缺失日期


# In[204]:


df_complete = df.set_index('交易日期').reindex(full_dates).reset_index()
df_complete.rename(columns={'index':'交易日期'}, inplace=True)

# 添加时间特征辅助插值
df_complete['星期'] = df_complete['交易日期'].dt.dayofweek + 1
df_complete['月份'] = df_complete['交易日期'].dt.month
df_complete['年度周数'] = df_complete['交易日期'].dt.isocalendar().week

# 方法2：线性插值（简单时间序列）
df_line = df_complete.copy()  # 改为英文命名避免混淆
df_line['销售量（克）_线性插值'] = df_line['销售量（克）'].interpolate(method='linear')

# 方法3：季节性分解插值
def seasonal_interpolate(series, period=5):
    """基于季节性分解的插值方法"""
    decomposed = seasonal_decompose(series.interpolate(method='linear'), period=period)
    seasonal = decomposed.seasonal
    trend = decomposed.trend
    resid = decomposed.resid
    
    # 用趋势+季节性填充缺失值
    reconstructed = trend + seasonal
    return np.where(np.isnan(series), reconstructed, series)

df_season = df_complete.copy()  # 改为英文命名避免混淆
df_season['销售量（克）_季节插值'] = seasonal_interpolate(df_season['销售量（克）'])

# 方法4：KNN插值（考虑星期特征）
df_knn = df_complete.copy()
imputer = KNNImputer(n_neighbors=5)
df_knn[['销售量（克）_KNN']] = imputer.fit_transform(df_knn[['销售量（克）','星期']])[:,0].reshape(-1,1)

# 合并所有插值结果（使用修正后的变量名）
method_mapping = {
    '线性插值': 'line',
    '季节插值': 'season',
    'KNN': 'knn'
}

for method_cn, method_en in method_mapping.items():
    df_complete[f'销售量（克）_{method_cn}'] = eval(f'df_{method_en}')[f'销售量（克）_{method_cn}']
df = df_complete


# In[206]:


df_complete


# In[208]:


df_complete = df_complete.drop(df.columns.difference(['交易日期','销售量（克）_线性插值']), axis=1)
df_complete.rename(columns={"销售量（克）_线性插值": "销售量（克）"}, inplace=True)
df = df_complete
df


# In[250]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import math
import warnings
warnings.filterwarnings('ignore')
# 1. 数据准备
# 读取Excel文件
# df = pd.read_excel('data.xlsx', sheet_name='Sheet1')

# 检查数据
print(df.head())
print(df.info())

# 处理重复日期（假设是数据录入错误，取平均值）
df = df.groupby('交易日期').mean().reset_index()

# 确保按日期排序
df = df.sort_values('交易日期')
df['交易日期'] = pd.to_datetime(df['交易日期'])


# 2. 数据预处理
# 使用销售量作为特征
data = df['销售量（克）'].values.reshape(-1, 1)

# 归一化
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# 创建数据集
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

# 选择时间窗口
look_back = 6
X, y = create_dataset(scaled_data, look_back)

# 划分训练集和测试集 (7:3)
train_size = int(len(X) * 0.7)
test_size = len(X) - train_size
X_train, X_test = X[0:train_size], X[train_size:len(X)]
y_train, y_test = y[0:train_size], y[train_size:len(y)]

# 重塑为LSTM输入格式 [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# 3. 构建LSTM模型
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 1)))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# 早停法
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# 训练模型
history = model.fit(X_train, y_train, 
                   validation_data=(X_test, y_test), 
                   epochs=500, 
                   batch_size=20, 
                   verbose=1
                   # ,callbacks=[early_stop]
                   )

# 4. 预测
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# 反归一化
train_predict = scaler.inverse_transform(train_predict)
y_train = scaler.inverse_transform([y_train])
test_predict = scaler.inverse_transform(test_predict)
y_test = scaler.inverse_transform([y_test])

# 5. 评估指标
def evaluate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f'MSE: {mse:.2f}')
    print(f'RMSE: {rmse:.2f}')
    print(f'MAE: {mae:.2f}')
    print(f'R2 Score: {r2:.2f}')
    
    return mse, rmse, mae, r2



# In[252]:


print("Train Metrics:")
train_metrics = evaluate_metrics(y_train[0], train_predict[:,0])

print("\nTest Metrics:")
test_metrics = evaluate_metrics(y_test[0], test_predict[:,0])


# In[254]:


###### # 6. 可视化
plt.figure(figsize=(15, 8))

# 训练损失
plt.subplot(2, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()

# 原始数据与预测数据对比
train_predict_plot = np.empty_like(data)
train_predict_plot[:, :] = np.nan
train_predict_plot[look_back:look_back+len(train_predict), :] = train_predict

test_predict_plot = np.empty_like(data)
test_predict_plot[:, :] = np.nan
test_predict_plot[look_back+len(train_predict):look_back+len(train_predict)+len(test_predict), :] = test_predict

plt.subplot(2, 1, 2)
plt.plot(df['交易日期'], data, label='Actual Data')
plt.plot(df['交易日期'], train_predict_plot, label='Training Prediction')
plt.plot(df['交易日期'], test_predict_plot, label='Testing Prediction')
plt.title('Sales Prediction')
plt.xlabel('Date')
plt.ylabel('Sales (g)')
plt.legend()

# 测试集详细对比
plt.figure(figsize=(15, 5))
test_dates = df['交易日期'][look_back+len(train_predict)+1:look_back+len(train_predict)+len(test_predict)+1]
plt.plot(test_dates, y_test[0], label='Actual Test Data')
plt.plot(test_dates, test_predict[:,0], label='Predicted Test Data')
plt.title('Test Set Prediction vs Actual')
plt.xlabel('Date')
plt.ylabel('Sales (g)')
plt.legend()

plt.tight_layout()
plt.show()


# In[ ]:




