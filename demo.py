import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
torch.manual_seed(1)

tsne = TSNE(n_components=2, perplexity=20, init='pca',learning_rate=1000)

output = torch.load('output.pt',map_location=torch.device('cpu'))
label = torch.load('emotion_label.pt',map_location=torch.device('cpu'))
x_tsne = tsne.fit_transform(output)


x_min, x_max = x_tsne.min(0), x_tsne.max(0)
x_tsne = (x_tsne-x_min) / (x_max - x_min)


markers = ['o'] * 7 + ['^'] * 7  # 前7个标签为圆形，后7个为三角形
# 设置颜色映射
colors = plt.cm.get_cmap('tab20', 14)  # 'tab20' 颜色映射支持最多20种颜色


# 绘制每个标签对应的点
for i in range(len(label.unique())):
    # 筛选出对应标签的数据
    idx = label == i
    plt.scatter(x_tsne[idx, 0] * 10000, x_tsne[idx, 1] * 10000, 
                c=[colors(i)],  # 设置颜色
                # marker=markers[i],  # 设置形状
                label=f'Class {i}',  # 图例标签
                edgecolors='black',  # 给点添加黑色边缘
                alpha=0.7)  # 设置透明度
# plt.scatter(x_tsne[:,0],x_tsne[:,1],c=label)
plt.show()
