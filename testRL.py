
#强化学习的策略网络模型定义+前向推理    还未更新网络权重 不算完整的训练过程

import torch
import torch.nn as nn

class robotpolicy(nn.Module):
    def __init__(self,obs_dim=6,action_dim=2):      #定义网络结构组成
        super().__init__()
        self.layer1=nn.Linear(obs_dim,32)
        self.layer2=nn.Linear(32,2)
        self.relu=nn.ReLU()

    def forward(self,obs):       #定义网络数据走向流程
        x=self.layer1(obs)
        x=self.relu(x)
        action=self.layer2(x)
        return action

model=robotpolicy()
obs=torch.tensor([0.5, 0.2, 0.1, -0.3, 0.8, 0.1])
action=model(obs)
print(action)





    














