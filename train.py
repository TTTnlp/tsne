import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

feature = torch.rand(2000, 256)
torch.save(feature, 'feature.pt')

domain_1 = torch.zeros(100, dtype=torch.long)
domain_2 = torch.ones(100, dtype=torch.long)
domain_label = torch.cat((domain_1, domain_2))
torch.save(domain_label, 'domain_label.pt')

emotion_1 = torch.randint(0,7,(1000,))
emotion_2 = torch.randint(7,14,(1000,))
emotion_label = torch.cat((emotion_1, emotion_2))
torch.save(emotion_label, 'emotion_label.pt')

class M(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.fc1 = nn.Linear(256,256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256,256)

        self.c = nn.Linear(256, 14)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self,x,y):
        x = self.fc1(x)  # 通过第一个全连接层
        x = self.relu(x)  # 激活函数
        output = self.fc2(x)  # 通过第二个全连接层

        x = self.c(output)

        loss = self.criterion(x,y)
        y = F.softmax(x, dim=-1)
        y = torch.argmax(y, dim=-1)
        return output,loss,y

feature = torch.load('feature.pt')
feature = feature.to('cuda')
# print(feature.shape)
emotion_label = torch.load('emotion_label.pt')

new_label = []
for l in emotion_label:
    temp_label = [0] * 14
    temp_label[l] = 1
    new_label.append(temp_label)
new_label = torch.tensor(new_label,dtype=torch.float32)
new_label = new_label.to('cuda')
# print(label)
model = M().to('cuda')


optimizer = optim.Adam(model.parameters(),lr=1e-3)
for e in range(200):
    model.train()
    _,loss,y = model(feature,new_label)
    print(f'第{e}个epoch的loss：为{loss}')
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 保存模型参数
torch.save(model.state_dict(), 'model.pth')
print("Model saved successfully!")



model.load_state_dict(torch.load('model.pth'))
model.eval()  # 切换到评估模式
with torch.no_grad():  # 禁用梯度计算
    # 假设我们使用相同的输入特征来做推理
    output, _, predicted_labels = model(feature, new_label)
    pred = predicted_labels.tolist()
    torch.save(output, 'output.pt')
correct = 0
for i,j in zip(emotion_label,pred):
    if i == j:
        correct += 1
print(correct / 2000)

print(emotion_label)
print(pred)
