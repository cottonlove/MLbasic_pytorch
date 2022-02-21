#DataLoad
#Using Dataset, DataLoader to perform minibatch 학습,shuffle, 병렬처리
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import TensorDataset # 텐서데이터셋
from torch.utils.data import DataLoader # 데이터로더

torch.manual_seed(1)

#Dataset 
x_train  =  torch.FloatTensor([[73,  80,  75], 
                               [93,  88,  93], 
                               [89,  91,  90], 
                               [96,  98,  100],   
                               [73,  66,  70]])  
y_train  =  torch.FloatTensor([[152],  [185],  [180],  [196],  [142]])

dataset = TensorDataset(x_train, y_train)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

#모델 초기화
model = nn.Linear(3,1)
#print(list(model.parameters()))
#optimizer 설정
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5) 

#학습
nb_epochs = 20
for epoch in range(nb_epochs + 1):
  for batch_idx, samples in enumerate(dataloader):
    # print(batch_idx)
    # print(samples)
    x_train, y_train = samples
    # H(x) 계산
    prediction = model(x_train)

    # cost 계산
    cost = F.mse_loss(prediction, y_train)

    # cost로 H(x) 계산
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    print('Epoch {:4d}/{} Batch {}/{} Cost: {:.6f}'.format(
        epoch, nb_epochs, batch_idx+1, len(dataloader),
        cost.item()
        ))
    
# 임의의 입력 [73, 80, 75]를 선언
new_var =  torch.FloatTensor([[73, 80, 75]]) 
# 입력한 값 [73, 80, 75]에 대해서 예측값 y를 리턴받아서 pred_y에 저장
pred_y = model(new_var) 
print("훈련 후 입력이 73, 80, 75일 때의 예측값 :", pred_y) 

############
#커스텀 데이터셋 (Custom Dataset)
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

torch.manual_seed(1)

#Dataset 상속
class CustomDataset(Dataset):
  def __init__(self):
    self.x_data = [[73, 80, 75],
                   [93, 88, 93],
                   [89, 91, 90],
                   [96, 98, 100],
                   [73, 66, 70]] #feature: 3개, sample: 5개
    self.y_data = [[152], [185], [180], [196], [142]]    
  # 총 데이터의 개수를 return
  def __len__(self):
    print("length")
    return len(self.x_data) 
  # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
  def __getitem__(self, idx):
    print("getitem")
    x = torch.FloatTensor(self.x_data[idx])
    #print('x is {}'.format(self.x_data[idx]))
    y = torch.FloatTensor(self.y_data[idx])        
    return x, y 

#dataset
print("dataset")
dataset = CustomDataset()
print('dataset length is {}'.format(len(dataset))) #5. 
dataloader = DataLoader(dataset, batch_size = 3, shuffle=True)
print("loader")

#model, optimizer 선언 및 초기화
model = torch.nn.Linear(3,1) #3개의 독립변수 x에서 하나의 y출력
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5) 

nb_epochs = 20
print("start learning")
for epoch in range(nb_epochs + 1):
  for batch_idx, samples in enumerate(dataloader):
    print("learning now")
    # print(batch_idx)
    # print(samples)
    x_train, y_train = samples
    # H(x) 계산
    prediction = model(x_train)

    # cost 계산
    cost = F.mse_loss(prediction, y_train)
    # cost로 H(x) 계산
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    

    # print('Epoch {:4d}/{} Batch {}/{}'.format(
    #     epoch, nb_epochs, batch_idx+1, len(dataloader),
    #     ))
    print('Epoch {:4d}/{} Batch {}/{} Cost: {:.6f}'.format(
        epoch, nb_epochs, batch_idx+1, len(dataloader),
        cost.item()
        ))
    
# 임의의 입력 [73, 80, 75]를 선언
new_var =  torch.FloatTensor([[73, 80, 75]]) 
# 입력한 값 [73, 80, 75]에 대해서 예측값 y를 리턴받아서 pred_y에 저장
pred_y = model(new_var) 
print("훈련 후 입력이 73, 80, 75일 때의 예측값 :", pred_y) 