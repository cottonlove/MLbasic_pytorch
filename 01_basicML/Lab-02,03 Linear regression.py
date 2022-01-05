#파이토치로 선형 회귀(linear regression) 구현
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#이 파이썬 코드를 재실행해도 다음에도 같은 결과가 나오도록 랜덤 시드(random seed)를 줍니다.
#torch.manual_seed()를 사용한 프로그램의 결과는 다른 컴퓨터에서 실행시켜도 동일한 결과를 얻을 수 있습니다. 
#그 이유는 torch.manual_seed()는 난수 발생 순서와 값을 동일하게 보장해준다는 특징때문입니다.
torch.manual_seed(1)

#1. 훈련 데이터 변수 선언
#보통은 불러옴
x_train = torch.FloatTensor([[1],[2],[3]])
#x_train = torch.FloatTensor([[1,2,3]])
y_train = torch.FloatTensor([[2],[4], [6]])

print(x_train)
print(x_train.shape)

#2. W, b 초기화
# 가중치 W를 0으로 초기화하고 학습을 통해 값이 변경되는 변수임을 명시함.
W = torch.zeros(1, requires_grad=True) 
print(W)
b = torch.zeros(1, requires_grad=True)
print(b)

#3. 가설 세우기
hypothesis = x_train * W + b
print(hypothesis) 

#4. 비용 함수 선언
cost = torch.mean((hypothesis - y_train) ** 2) 
print(cost)

#5. 경사 하강법 구현
#SGD: 경사 하강법의 일종
optimizer = optim.SGD([W,b], lr = 0.01)

# gradient를 0으로 초기화
optimizer.zero_grad() 
# 비용 함수를 미분하여 gradient 계산
cost.backward() 
# W와 b를 업데이트
optimizer.step() 

#######################
#전체 코드

#데이터
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])

#모델 초기화
W = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

#optimizer 설정
optimizer = optim.SGD([W, b], lr=0.01)

nb_epochs = 2000 # 원하는만큼 경사 하강법을 반복
for epoch in range(nb_epochs + 1):

    # H(x) 계산
    hypothesis = x_train * W + b

    # cost 계산
    cost = torch.mean((hypothesis - y_train) ** 2)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 100번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} W: {:.3f}, b: {:.3f} Cost: {:.6f}'.format(
            epoch, nb_epochs, W.item(), b.item(), cost.item()
        ))

###########
#자동 미분(Autograd)
'''경사 하강법 코드를 보고있으면 requires_grad=True, backward() 등이 나옵니다.
 이는 파이토치에서 제공하고 있는 자동 미분(Autograd) 기능을 수행하고 있는 것'''

import torch
#값이 2인 스칼라 텐서 w 선언
# required_grad를 True로 설정합니다. 이는 이 텐서에 대한 기울기를 저장하겠다는 의미 -> w.grad에 w에 대한 미분값이 저장됨
w = torch.tensor(2.0, requires_grad=True)
y = w**2
z = 2*y + 5
#backward()를 호출하면 해당 수식의 w에 대한 기울기를 계산
z.backward()
#w.grad를 출력하면 w가 속한 수식을 w로 미분한 값이 저장된 것을 확인
print('수식을 w로 미분한 값 : {}'.format(w.grad))

###########
#다중 선형 회귀(Multivariable Linear regression)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

# 훈련 데이터
x1_train = torch.FloatTensor([[73], [93], [89], [96], [73]])
x2_train = torch.FloatTensor([[80], [88], [91], [98], [66]])
x3_train = torch.FloatTensor([[75], [93], [90], [100], [70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

# 가중치w와 bias b 선언
# x가 3개이므로 w도 3개 선언
w1 = torch.zeros(1, requires_grad=True)
w2 = torch.zeros(1, requires_grad=True)
w3 = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# optimizer 설정
optimizer = optim.SGD([w1,w2,w3,b], lr = 1e-5)

nb_epochs = 1000
for epoch in range(nb_epochs+1):
  #H(x) 계산
  hypothesis = x1_train * w1 + x2_train * w2 + x3_train * w3 + b
  #cost 계산
  cost = torch.mean((hypothesis - y_train) ** 2)
  # cost로 H(x) 개선
  optimizer.zero_grad()
  cost.backward()
  optimizer.step()

 # 100번마다 로그 출력
  if epoch % 100 == 0:
      print('Epoch {:4d}/{} w1: {:.3f} w2: {:.3f} w3: {:.3f} b: {:.3f} Cost: {:.6f}'.format(
          epoch, nb_epochs, w1.item(), w2.item(), w3.item(), b.item(), cost.item()
      ))

###########
#x의 갯수가 많아지면 위의 방식은 비효율적 -> 가설을 벡터와 행렬 연산으로 표현
#훈련 데이터
x_train  =  torch.FloatTensor([[73,  80,  75], 
                               [93,  88,  93], 
                               [89,  91,  80], 
                               [96,  98,  100],   
                               [73,  66,  70]])  
y_train  =  torch.FloatTensor([[152],  [185],  [180],  [196],  [142]])

# 모델 초기화
W = torch.zeros((3, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
# optimizer 설정
optimizer = optim.SGD([W, b], lr=1e-5)

nb_epochs = 20
for epoch in range(nb_epochs+1):
  #H(x) 계산
  # 편향 b는 broadcasting되어서 각 샘플에 더해진다.
  hypothesis = x_train.matmul(W) + b

  #cost 계산
  cost = torch.mean((hypothesis-y_train)**2)

  #cost로 H(x) 개선
  optimizer.zero_grad()
  cost.backward()
  optimizer.step()

  print('Epoch {:4d}/{} hypothesis: {} Cost: {:.6f}'.format(
          epoch, nb_epochs, hypothesis.squeeze().detach(), cost.item()
      ))

##########
#nn.Module로 구현하는 선형 회귀
#파이토치에서 이미 구현되어져 제공되고 있는 함수들을 불러오는 것
#파이토치에서는 선형 회귀 모델이 nn.Linear()라는 함수로, 또 평균 제곱오차가 nn.functional.mse_loss()라는 함수로 구현

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)

#data 선언
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])

# 모델을 선언 및 초기화. 단순 선형 회귀이므로 input_dim=1, output_dim=1.
model = nn.Linear(1,1)

print(list(model.parameters())) #W,b가 랜덤 초기화되어 있음
# optimizer 설정. 경사 하강법 SGD를 사용하고 learning rate를 의미하는 lr은 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=0.01) 

# 전체 훈련 데이터에 대해 경사 하강법을 2,000회 반복
nb_epochs = 2000
for epoch in range(nb_epochs+1):

    # H(x) 계산
    prediction = model(x_train)

    # cost 계산
    cost = F.mse_loss(prediction, y_train) # <== 파이토치에서 제공하는 평균 제곱 오차 함수

    # cost로 H(x) 개선하는 부분
    # gradient를 0으로 초기화
    optimizer.zero_grad()
    # 비용 함수를 미분하여 gradient 계산
    cost.backward() # backward 연산
    # W와 b를 업데이트
    optimizer.step()

    if epoch % 100 == 0:
    # 100번마다 로그 출력
      print('Epoch {:4d}/{} Cost: {:.6f}'.format(
          epoch, nb_epochs, cost.item()
      ))

#학습된 모델 이용하여 예측
# 임의의 입력 4를 선언
new_var =  torch.FloatTensor([[4.0]]) 
# 입력한 값 4에 대해서 예측값 y를 리턴받아서 pred_y에 저장
pred_y = model(new_var) # forward 연산
# y = 2x 이므로 입력이 4라면 y가 8에 가까운 값이 나와야 제대로 학습이 된 것
print("훈련 후 입력이 4일 때의 예측값 :", pred_y) 
#학습 후 W,b 출력
print(list(model.parameters()))

##########
#다중 선형 회귀 구현
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)

# 데이터
#데이터 선언
#3개의 x로부터 하나의 y예측. 5개의 sample, 3개의 feature
x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

#모델 선언 및 초기화
model = nn.Linear(3,1) #3개의 x로부터 하나의 y예측
print(list(model.parameters())) #W,b가 랜덤 초기화되어 있음
# optimizer 설정. 경사 하강법 SGD를 사용하고
#학습률(learning rate)은 0.00001로 정합니다. 파이썬 코드로는 1e-5로도 표기합니다. 0.01로 하지 않는 이유는 기울기가 발산하기 때문
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5) 

# 전체 훈련 데이터에 대해 경사 하강법을 2,000회 반복
nb_epochs = 2000
for epoch in range(nb_epochs+1):

    # H(x) 계산
    prediction = model(x_train)

    # cost 계산
    cost = F.mse_loss(prediction, y_train) # <== 파이토치에서 제공하는 평균 제곱 오차 함수

    # cost로 H(x) 개선하는 부분
    # gradient를 0으로 초기화
    optimizer.zero_grad()
    # 비용 함수를 미분하여 gradient 계산
    cost.backward() # backward 연산
    # W와 b를 업데이트
    optimizer.step()

    if epoch % 100 == 0:
    # 100번마다 로그 출력
      print('Epoch {:4d}/{} Cost: {:.6f}'.format(
          epoch, nb_epochs, cost.item()
      ))

#학습된 모델 이용하여 예측
# 임의의 입력 [73, 80, 75]를 선언
new_var =  torch.FloatTensor([[73, 80, 75]]) 
# 입력한 값 [73, 80, 75]에 대해서 예측값 y를 리턴받아서 pred_y에 저장
pred_y = model(new_var) 
print("훈련 후 입력이 73, 80, 75일 때의 예측값 :", pred_y) 

#학습 후 3개의 w와 b의 값 출력
print(list(model.parameters()))

###########
#단순 선형 회귀 클래스 class 로 구현하기
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)

# 데이터
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])

#class로 구현
class LinearRegressionModel(nn.Module): #torch.nn.Module을 상속받는 파이썬 클래스
  def __init__(self):
    super().__init__() #super() 함수를 부르면 여기서 만든 클래스는 nn.Module 클래스의 속성들을 가지고 초기화 됩니다
    self.linear = nn.Linear(1,1)
  
  def forward(self, x):
    return self.linear(x)

#모델 선언
model = LinearRegressionModel()
print(list(model.parameters()))

# optimizer 설정. 경사 하강법 SGD를 사용하고 learning rate를 의미하는 lr은 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=0.01) 
# 전체 훈련 데이터에 대해 경사 하강법을 2,000회 반복
nb_epochs = 2000
for epoch in range(nb_epochs+1):

    # H(x) 계산
    prediction = model(x_train)

    # cost 계산
    cost = F.mse_loss(prediction, y_train) # <== 파이토치에서 제공하는 평균 제곱 오차 함수

    # cost로 H(x) 개선하는 부분
    # gradient를 0으로 초기화
    optimizer.zero_grad()
    # 비용 함수를 미분하여 gradient 계산
    cost.backward() # backward 연산
    # W와 b를 업데이트
    optimizer.step()

    if epoch % 100 == 0:
    # 100번마다 로그 출력
      print('Epoch {:4d}/{} Cost: {:.6f}'.format(
          epoch, nb_epochs, cost.item()
      ))
#학습된 모델로 예측
new_val =  torch.FloatTensor([[4.0]])
pred_y = model(new_val) 
print("훈련 후 입력이 4.0일 때의 예측값 :", pred_y)

########
#다중 선형 회귀 클래스 class 로 구현
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)

# 데이터
x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

#class로 모델 선언
class MultivariateLinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1) # 다중 선형 회귀이므로 input_dim=3, output_dim=1.

    def forward(self, x):
        return self.linear(x)
#모델 선언 및 초기화
model = MultivariateLinearRegressionModel()
print(list(model.parameters()))
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5) 

nb_epochs = 2000
for epoch in range(nb_epochs+1):

    # H(x) 계산
    prediction = model(x_train)
    # model(x_train)은 model.forward(x_train)와 동일함.

    # cost 계산
    cost = F.mse_loss(prediction, y_train) # <== 파이토치에서 제공하는 평균 제곱 오차 함수

    # cost로 H(x) 개선하는 부분
    # gradient를 0으로 초기화
    optimizer.zero_grad()
    # 비용 함수를 미분하여 gradient 계산
    cost.backward()
    # W와 b를 업데이트
    optimizer.step()

    if epoch % 100 == 0:
    # 100번마다 로그 출력
      print('Epoch {:4d}/{} Cost: {:.6f}'.format(
          epoch, nb_epochs, cost.item()
      ))
