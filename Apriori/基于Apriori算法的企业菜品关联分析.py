import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dish = pd.read_csv('../data/meal_dishes_detail.csv',encoding='utf-8')
info = pd.read_csv('../data/meal_order_info.csv',encoding='utf-8')
detail = pd.read_csv('../data/meal_order_detail.csv',encoding='utf-8')

info_id = info[['info_id']][(info.order_status==0) | (info.order_status==2)]["info_id"].tolist()
proportion = len(info_id)/info.shape[0]

info_1 = info[info['order_status'].isin([1])]
info_1 = info_1.reset_index(drop=True)
info_1.to_csv('../tmp/info.csv',encoding='utf-8')

#把原来转换好的日期格式，重新赋值回这一列
#把原来的字符串时间，改为纯日期格式，方便后续做时间筛选，计算
# 正确、干净、高效、不报错
info_1['use_start_time'] = pd.to_datetime(info_1['use_start_time']).dt.date

#统计每日用餐人数和营业额
groupbyday = info_1[['use_start_time','number_consumers','accounts_payable']].groupby(by='use_start_time')
sale_day = groupbyday.sum()

#写出每日用餐人数和营业额
sale_day.columns = ['人数','销量']
sale_day.to_csv('../tmp/sale_day.csv',encoding='utf-8_sig')
print('每日的用餐人数和营业额：\n',sale_day.head(10))

# 每日用餐人数和营业额折线图
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
fig, ax1 = plt.subplots()  # 使用subplots函数创建窗口
ax1.plot(sale_day['人数'], '--')
ax1.set_yticks(range(0, 900, 100))  # 设置y轴的刻度范围
ax1.legend(('用餐人数',), loc='upper left', fontsize=10)
ax2 = ax1.twinx()  # 创建第二个坐标轴
ax2.plot(sale_day['销量'])
ax2.legend(('营业额',), loc='upper right', fontsize=10)
ax1.set_xlabel('日期')
ax1.set_ylabel('用餐人数')
ax2.set_ylabel('营业额（元）')
plt.gcf().autofmt_xdate()  # 自动适应刻度线密度，包括x轴、y轴
plt.title('每日用餐人数和营业额')
plt.show()


