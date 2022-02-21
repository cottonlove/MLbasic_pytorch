'''
Rogistic Regression: 이진 분류 수행하는 모델

둘 중하나를 결정하는 Binary Classification을 위한 대표적인 알고리즘
'''
#import
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#for reporducibility (다른 곳에서 돌려도 같은 결과 나오게)
torch.manual_seed(1)

#Training Data
x_data = [[1,2],[2,3],[3,1],[4,3],[5,3],[6,2]] #|x_data| = (6,2)  m=6, d=2
y_data = [[0], [0],[0], [1],[1],[1]] #|y_data| = (6,)

#array -> torch.Tensor format으로 바꾸기!!!
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

print(x_train.shape)
print(y_train.shape)

#가중치 초기화 및 선언
W = torch.zeros((2,1), requires_grad=True)
b = torch.zeros(1, requires_grad = True)

#hypothesis
'''hypothesis = 1/(1+torch.exp(-(x_train.matmul(W) + b)))'''
hypothesis = torch.sigmoid(x_train.matmul(W) + b)
#x.matmul(w) = torch.matmul(x,w)
#print(hypothesis)
print(hypothesis.shape)

#Computing Cost function
'''
#오차
losses = -(y_train * torch.log(hypothesis)+(1-y_train)* toch.log(1-hypothesis))
print(losses)
#전체 오차 평균
cost = losses.mean()
지금까지 비용 함수의 값을 직접 구현하였는데, 사실 파이토치에서는 로지스틱 회귀의 비용 함수를 이미 구현해서 제공하고 있습니다.
'''
F.binary_cross_entropy(hypothesis, y_train)

'''
Whole Training Procedure
'''
#optimizer설정
optimizer = optim.SGD([W,b], lr = 1)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):
  #Cost 계산
  hypothesis = torch.sigmoid(x_train.matmul(W) + b)
  cost = F.binary_cross_entropy(hypothesis, y_train)

  #Cost로 H(x)=P(y=1; W)개선
  optimizer.zero_grad() #gradeient초기화 꼭 하기
  cost.backward()
  optimizer.step()

  #100번마다 로그 출력
  if epoch % 100 == 0:
    print('Epoch {:4d}/{} Cost: {:.6f}'.format(
        epoch, nb_epochs, cost.item()
    ))
'''
Evaluation

hypothesis = p(y=1 ; W)
'''
hypothesis = torch.sigmoid(x_train.matmul(W) + b)
print(hypothesis)

'''
현재 위 값들은 0과 1 사이의 값을 가지고 있습니다. 이제 0.5를 넘으면 True, 넘지 않으면 False로 값을 정하여 출력해보겠습니다.
'''
prediction = hypothesis >= torch.FloatTensor([0.5])
print(prediction)

correct_prediction = prediction.float() == y_train
print(correct_prediction)

print("After Training, W is {} b is {}".format(W,b))

'''
위와 같은 클래스를 사용한 모델 구현 형식은 대부분의 파이토치 구현체에서 사용하고 있는 방식으로 반드시 숙지할 필요가 있습니다.

클래스(class) 형태의 모델은 nn.Module 을 상속받습니다. 그리고 __init__()에서 모델의 구조와 동적을 정의하는 생성자를 정의합니다. 이는 파이썬에서 객체가 갖는 속성값을 초기화하는 역할로, 객체가 생성될 때 자동으호 호출됩니다. super() 함수를 부르면 여기서 만든 클래스는 nn.Module 클래스의 속성들을 가지고 초기화 됩니다. 

foward() 함수는 모델이 학습데이터를 입력받아서 forward 연산을 진행시키는 함수입니다. 이 forward() 함수는 model 객체를 데이터와 함께 호출하면 자동으로 실행이됩니다. 예를 들어 model이란 이름의 객체를 생성 후, model(입력 데이터)와 같은 형식으로 객체를 호출하면 자동으로 forward 연산이 수행됩니다.

H(x) 식에 입력 로부터 예측된 를 얻는 것을 forward 연산이라고 합니다.
'''

class BinaryClassifier(nn.Module):
  def __init__(self):
    super().__init__()
    self.linear = nn.Linear(2,1) #input_dim = 2, output_dim = 1 <-> |x|=(?,2) |W| = (2,1), |b| = (1,)
    self.sigmoid = nn.Sigmoid()
  
  def forward(self, x):
    return self.sigmoid(self.linear(x))

#model 선언
model = BinaryClassifier()

#optimizer선언
optimizer = optim.SGD(model.parameters(), lr = 1) #model.parameters()를 통해, W, b가 iterator형식으로 들어옴

nb_epochs = 100
for epoch in range(nb_epochs + 1):
  #H(x) = P(y=1; W) 계산
  hypothesis = model(x_train)

  #cost 계산
  cost = F.binary_cross_entropy(hypothesis, y_train)

  #cost로 H(x) 개선
  optimizer.zero_grad()
  cost.backward()
  optimizer.step()

  #20번마다 로그 출력
  if epoch % 10 == 0:
        prediction = hypothesis >= torch.FloatTensor([0.5]) # 예측값이 0.5를 넘으면 True로 간주
        correct_prediction = prediction.float() == y_train # 실제값과 일치하는 경우만 True로 간주
        accuracy = correct_prediction.sum().item() / len(correct_prediction) # 정확도를 계산
        print('Epoch {:4d}/{} Cost: {:.6f} Accuracy {:2.2f}%'.format( # 각 에포크마다 정확도를 출력
            epoch, nb_epochs, cost.item(), accuracy * 100,
        ))

'''
nn.Module로 구현하는 Logistic Regression
'''
model = nn.Sequential(nn.Linear(2,1), # input_dim = 2, output_dim = 1
                      nn.Sigmoid() # 출력은 시그모이드 함수를 거친다
                      )

model(x_train) #학습 전 예측
# optimizer 설정
optimizer = optim.SGD(model.parameters(), lr=1)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):

    # H(x) 계산
    hypothesis = model(x_train)

    # cost 계산
    cost = F.binary_cross_entropy(hypothesis, y_train)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 20번마다 로그 출력
    if epoch % 10 == 0:
        prediction = hypothesis >= torch.FloatTensor([0.5]) # 예측값이 0.5를 넘으면 True로 간주
        correct_prediction = prediction.float() == y_train # 실제값과 일치하는 경우만 True로 간주
        accuracy = correct_prediction.sum().item() / len(correct_prediction) # 정확도를 계산
        print('Epoch {:4d}/{} Cost: {:.6f} Accuracy {:2.2f}%'.format( # 각 에포크마다 정확도를 출력
            epoch, nb_epochs, cost.item(), accuracy * 100,
        ))

model(x_train) #학습 후 예측
#학습 후 최적화된 가중치 출력
print(list(model.parameters()))