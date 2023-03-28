import pandas as pd
from sklearn.model_selection import train_test_split

# 读取CSV文件
data = pd.read_csv('ISIC_2019_train.csv')

# 随机分成训练集和测试集，比例为7:3
train_data, test_data = train_test_split(data, test_size=0.3)

# 保存训练集和测试集到CSV文件
train_data.to_csv('train_data.csv', index=False)
test_data.to_csv('test_data.csv', index=False)