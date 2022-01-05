# Minibatch Gradient Descent (미니배치 경사하강법)

많은 양의 데이터를 한번에 학습시키는 것은 불가능 -> 일부분의 데이터로만 학습

* 전체 데이터를 균일하게 나눠서 학습하자!

## 미니 배치와 배치크기(Mini Batch and Batch Size)

![image-20220105204218034](C:\Users\dbw21\AppData\Roaming\Typora\typora-user-images\image-20220105204218034.png)

미니 배치 학습을 하게되면 미니 배치만큼만 가져가서 미니 배치에 대한 대한 비용(cost)를 계산하고, 경사 하강법을 수행합니다. 그리고 다음 미니 배치를 가져가서 경사 하강법을 수행하고 마지막 미니 배치까지 이를 반복합니다. 이렇게 전체 데이터에 대한 학습이 1회 끝나면 1 에포크(Epoch)가 끝나게 됩니다.

* **에포크(Epoch)는 전체 훈련 데이터가 학습에 한 번 사용된 주기**

미니 배치 학습에서는 미니 배치의 개수만큼 경사 하강법을 수행해야 전체 데이터가 한 번 전부 사용되어 1 에포크(Epoch)가 됩니다. 미니 배치의 개수는 결국 미니 배치의 크기를 몇으로 하느냐에 따라서 달라지는데 미니 배치의 크기를 배치 크기(batch size)라고 합니다.

* 배치 경사 하강법: 전체 데이터에 대해서 한 번에 경사 하강법을 수행하는 방법
* 미니 배치 경사 하강법: 미니 배치 단위로 경사 하강법을 수행하는 방법
* 배치 경사 하강법은 경사 하강법을 할 때, 전체 데이터를 사용하므로 가중치 값이 최적값에 수렴하는 과정이 매우 안정적이지만, 계산량이 너무 많이 듭니다. 미니 배치 경사 하강법은 경사 하강법을 할 때, 전체 데이터의 일부만을 보고 수행하므로 최적값으로 수렴하는 과정에서 값이 조금 헤매기도 하지만 훈련 속도가 빠릅니다.
* 배치 크기는 보통 2의 제곱수를 사용합니다. ex) 2, 4, 8, 16, 32, 64... 그 이유는 CPU와 GPU의 메모리가 2의 배수이므로 배치크기가 2의 제곱수일 경우에 데이터 송수신의 효율을 높일 수 있다고 합니다.



## 이터레이션(Iteration)

![image-20220105205151390](C:\Users\dbw21\AppData\Roaming\Typora\typora-user-images\image-20220105205151390.png)

* Iteration: 한번의 epoch내에서 이루어지는 매개변수인 가중치 W와 b의 업데이트 횟수

전체 데이터가 2,000일때, batch size가 200이라면 이터레이션 수는 총 10개. 즉 한 번의 epoch당 매개변수 업데이트가 10번 이루어짐을 의미한다.



## Data Load (데이터 로드하기)

파이토치에서는 데이터를 좀 더 쉽게 다룰 수 있도록 유용한 도구로서 데이터셋(Dataset)과 데이터로더(DataLoader)를 제공하고 이를 사용하면 **미니 배치 학습**, 데이터 셔플(shuffle), 병렬 처리까지 간단히 수행가능.

기본적인 사용 방법은 Dataset을 정의하고, 이를 DataLoader에 전달하는 것이다. 

Dataset을 커스텀하여 만들 수도 있지만 여기서는 텐서를 입력받아 Dataset의 형태로 변환해주는 **TensorDataset**을 사용해보겠습니다.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset # 텐서데이터셋
from torch.utils.data import DataLoader # 데이터로더
```

TensorDataset은 기본적으로 텐서를 입력으로 받습니다. 텐서 형태로 데이터를 정의합니다.

```python
x_train  =  torch.FloatTensor([[73,  80,  75], 
                               [93,  88,  93], 
                               [89,  91,  90], 
                               [96,  98,  100],   
                               [73,  66,  70]])  
y_train  =  torch.FloatTensor([[152],  [185],  [180],  [196],  [142]])
```

이제 이를 TensorDataset의 입력으로 사용하고 dataset으로 저장합니다.

```python
dataset = TensorDataset(x_train, y_train)
```

파이토치의 데이터셋을 만들었다면 데이터로더를 사용 가능합니다. **데이터로더는 기본적으로 2개의 인자**를 입력받습니다. 하나는 **데이터셋, 미니 배치의 크기**입니다. 이때 미니 배치의 크기는 통상적으로 2의 배수를 사용합니다. (ex) 64, 128, 256...) 그리고 **추가적으로 많이 사용되는 인자로 shuffle**이 있습니다. shuffle=True를 선택하면 Epoch마다 데이터셋을 섞어서 데이터가 학습되는 순서를 바꿉니다.모델이 데이터셋의 순서에 익숙해지는 것을 방지하여 학습할 때는 이 옵션을 True를 주는 것을 권장합니다. 

```python
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
```

```python
model = nn.Linear(3,1)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5) 
nb_epochs = 20
for epoch in range(nb_epochs + 1):
  for batch_idx, samples in enumerate(dataloader): 
    # enumerate(dataloader): minibatch 인덱스와 데이터를 받음.
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
        )) #len(dataloader): 한 epoch당 minibatch 개수

```



* enumerate(dataloader): minibatch 인덱스와 데이터를 받음.
* len(dataloader) : 한 epoch당 minibatch 개수

- [ ] **Q.** 이런식으로 한 epoch내 iteration에 따라 cost값의 변동이 있는데, 그러면 제대로 학습이 잘 되고 있는 지 알 수 있는 방법은? 보통 cost가 줄어들면 lr이 잘 설정된 거고 학습이 잘 된 걸 알 수 있는데,,,

-> epoch수를 20에서 100으로 늘려도 cost의 대체적 값이 크게 감소하지 않았는데 그렇다는 것은 data양이 적어서 그런것인가..? 이걸 알기위해 validation dataset이 필요한 건가..?

![image-20220105212028699](C:\Users\dbw21\AppData\Roaming\Typora\typora-user-images\image-20220105212028699.png)



## 커스텀 데이터셋

torch.utils.data.Dataset을 상속받아 직접 커스텀 데이터셋(Custom Dataset)을 만드는 경우도 있습니다. torch.utils.data.Dataset은 파이토치에서 데이터셋을 제공하는 추상 클래스입니다. Dataset을 상속받아 다음 메소드들을 오버라이드 하여 커스텀 데이터셋을 만들어보겠습니다.

커스텀 데이터셋을 만들 때, 일단 가장 기본적인 뼈대는 아래와 같습니다. 여기서 필요한 기본적인 define은 3개입니다.



```python
class CustomDataset(torch.utils.data.Dataset): 
  def __init__(self):
  데이터셋의 전처리를 해주는 부분

  def __len__(self):
  데이터셋의 길이. 즉, 총 샘플의 수를 적어주는 부분

  def __getitem__(self, idx): 
  데이터셋에서 특정 1개의 샘플을 가져오는 함수
```

- len(dataset)을 했을 때 데이터셋의 크기를 리턴할 **len**
- dataset[i]을 했을 때 i번째 샘플을 가져오도록 하는 인덱싱을 위한 **get_item**



### 커스텀 데이터셋 (Custom Dataset)으로 linear regression (선형회귀) 구현하기

```python
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# Dataset 상속
class CustomDataset(Dataset): 
  def __init__(self):
    self.x_data = [[73, 80, 75],
                   [93, 88, 93],
                   [89, 91, 90],
                   [96, 98, 100],
                   [73, 66, 70]]
    self.y_data = [[152], [185], [180], [196], [142]]

  # 총 데이터의 개수를 리턴
  def __len__(self): 
    return len(self.x_data) #5

  # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
  def __getitem__(self, idx): 
    x = torch.FloatTensor(self.x_data[idx])
    y = torch.FloatTensor(self.y_data[idx])
    return x, y

dataset = CustomDataset()
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

#모델 및 optimizer선언
model = torch.nn.Linear(3,1)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5) 

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
```



- [ ] Q.  len이랑 getitem 언제 call되는건지...? ->  print로 찍어봤을 때는

```python
dataset = CustomDataset() #len 1번
dataloader = DataLoader(dataset, batch_size = 2, shuffle=True) #len 2번 -> 왜???

for batch_idx, samples in enumerate(dataloader): #len 1번, getitem2번(batchsize=2라서)
    #batchsize가 2이고 len(dataset) = 5이므로 2.2.1로 나눠진 거라 마지막은 getitem1번만 불린다.
```

```
# 총 데이터의 개수를 리턴
  def __len__(self): 
    return len(self.x_data) #5

  # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
  def __getitem__(self, idx): 
    x = torch.FloatTensor(self.x_data[idx])
    y = torch.FloatTensor(self.y_data[idx])
    return x, y
```

