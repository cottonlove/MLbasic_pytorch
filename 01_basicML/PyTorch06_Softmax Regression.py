import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#훈련 데이터와 레이블을 텐서로 선언합니다.
#array -> Tensor
x_train = [[1, 2, 1, 1],
           [2, 1, 3, 2],
           [3, 1, 3, 4],
           [4, 1, 5, 5],
           [1, 7, 5, 5],
           [1, 2, 5, 6],
           [1, 6, 6, 6],
           [1, 7, 7, 7]] #|x_train|= (8, 4). #of samples = 8 . #of features = 4
y_train = [2, 2, 2, 1, 1, 1, 0, 0] #|y_train| = (8,) 
#y_train은 각 샘플에 대한 레이블인데, 여기서는 0, 1, 2의 값을 가지는 것으로 보아 총 3개의 클래스가 존재
x_train = torch.FloatTensor(x_train)
y_train = torch.LongTensor(y_train)

# 모델 초기화
W = torch.zeros((4, 3), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
# optimizer 설정
optimizer = optim.SGD([W, b], lr=0.1)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):

  #Cost 계산
  z = x_train.matmul(W) + b #hypotheis
  #주의할 점은 F.cross_entropy()는 그 자체로 소프트맥스 함수를 포함하고 있으므로 
  #가설에서는 소프트맥스 함수를 사용할 필요가 없습니다.
  cost = F.cross_entropy(z, y_train) #원-핫 벡터를 넣을 필요없이 바로 실제값을 인자로 사용
  # cost로 H(x) 개선
  optimizer.zero_grad()
  cost.backward()
  optimizer.step()

  # 100번마다 로그 출력
  if epoch % 100 == 0:
      print('Epoch {:4d}/{} Cost: {:.6f}'.format(
          epoch, nb_epochs, cost.item()
      ))

## 소프트맥스 회귀 nn.Module로 구현하기
# 모델을 선언 및 초기화. 4개의 features(특성)을 가지고 3개의 클래스로 분류. input_dim=4, output_dim=3.
model = nn.Linear(4, 3)

#아래에서 F.cross_entropy()를 사용할 것이므로 따로 소프트맥스 함수를 가설에 정의하지 않습니다.
# optimizer 설정
optimizer = optim.SGD(model.parameters(), lr=0.1)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):

    # H(x) 계산
    prediction = model(x_train)

    # cost 계산
    cost = F.cross_entropy(prediction, y_train)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 20번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))

## 소프트맥스 회귀 클래스로 구현하기
class SoftmaxClassifierModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 3) # Outputdim이 3!

    def forward(self, x):
        return self.linear(x)

model = SoftmaxClassifierModel()

# optimizer 설정
optimizer = optim.SGD(model.parameters(), lr=0.1)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):

    # H(x) 계산
    prediction = model(x_train)

    # cost 계산
    cost = F.cross_entropy(prediction, y_train)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 20번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))