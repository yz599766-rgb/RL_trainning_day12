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


model=Net()

optimiser=torch.optim.Adam(model.parameters(),lr=1e-3)


x=torch.linspace(-3.14,3.14,100).unsqueeze(1)
y_true=torch.sin(x)


for i in range(1000):
    y_pred=model(x)
    loss_fn=nn.MSELoss()
    loss=loss_fn(y_pred,y_true)

    optimiser.zero_grad()
    loss.backward()
    optimiser.step()

    if i%100==0:
        print(f"第{i}轮， loss={loss.item():.4f}")


test_x=torch.tensor([[1.57]])
print(model(test_x))






