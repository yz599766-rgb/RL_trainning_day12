#两层网络完整拟合sin代码   强化学习训练全流程

import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1=nn.Linear(1,32)
        self.layer2=nn.Linear(32,1)


    def forward(self,x):
        x=self.layer1(x)
        x=torch.relu(x)
        x=self.layer2(x)
        return x

model =Net()           #我创建的一个实例网络    后续为我工作的网络模型叫做model   


optimiser=torch.optim.Adam(model.parameters(),lr=1e-3)     #创建实例模型后  选择优化器

#准备数据
x=torch.linspace(-3.14,3.14,100).unsqueeze(1)
y_true=torch.sin(x)

#准备完数据后开始循环  
for i in range(1000):    #加上循环，训练1000次       loss不断下降 网络学会拟合sin       不加循环 只训练一次 网络仅微调一丢丢
    y_pred=model(x)
    loss_fn=nn.MSELoss()     #创建打分器loss_fn   
    loss=loss_fn(y_true,y_pred)


    #loss一算出来 开始优化
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()

    if i%100==0:
        print(f"第{i}轮，loss={loss.item():.4f}")           #loss不断下降  准确度越来越高  这个sin拟合模型modle很厉害了
     #`.item()` 的作用：**把 tensor 里面存的数字单独抠出来，变成 Python 普通浮点数**。



#model具备拟合sin的能力  可以输入x 预测出sinx的值了  测试
test_x=torch.tensor([[1.57]])   #两个中括号 二维  一行以样本  一列一特征   nn.Linear就认这种表格形式
print(model(test_x))
        



















