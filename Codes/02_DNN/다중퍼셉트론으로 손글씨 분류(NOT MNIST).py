#데이터 확인
'''
load_digits()를 통해 이미지 데이터를 로드할 수 있습니다. 로드한 전체 데이터를 digits에 저장합니다.
'''
%matplotlib inline
import matplotlib.pyplot as plt # 시각화를 위한 맷플롯립
from sklearn.datasets import load_digits
digits = load_digits() # 1,979개의 이미지 데이터 로드

#첫번째 샘플을 출력해보겠습니다.
#images[인덱스]를 사용하면 해당 인덱스의 이미지를 행렬로서 출력할 수 있습니다.
print(digits.images[0])
print(digits.target[0])

#샘플 개수 확인
print("전체 샘플 개수: {}".format(len(digits.images)))

#전체 샘플 중에서 상위 5개의 샘플만 시각화해봅시다.
images_and_labels = list(zip(digits.images, digits.target))
#print(images_and_labels[0])
for index, (image,label) in enumerate(images_and_labels[:5]): #0-4. 5개의 샘플 출력
  plt.subplot(2, 5, index+1)
  plt.axis('off')
  plt.imshow(image, cmap=plt.cm.gray_r, interpolation = 'nearest')
  plt.title('sample: %i' %label)

print(digits.data[0])

X = digits.data #이미지 8*8=64차원의 벡터
Y = digits.target #각 이미지에 대한 label(0-9)
print("length of X is {}".format(len(X)))
print("Shape of X is {}".format(X.shape))
print("length of Y is {}".format(len(Y)))

#다층 퍼셉트론 분류기 만들기
import torch
import torch.nn as nn
from torch import optim

# for reproducibility
torch.manual_seed(777)

model = nn.Sequential(
    nn.Linear(64, 32), # input_layer = 64, hidden_layer1 = 32
    nn.ReLU(),
    nn.Linear(32,16), # hidden_layer2 = 32, hidden_layer3 = 16
    nn.ReLU(),
    nn.Linear(16,10) # hidden_layer3 = 16, output_layer = 10
    #crossEntropy 비용함수는 softmax함수를 포함하고 있음
)

#torchTensor로 바꾸기
X = torch.tensor(X, dtype=torch.float32)
Y = torch.tensor(Y, dtype=torch.int64)

#비용함수와 옵티마이저 설정
loss_fn = nn.CrossEntropyLoss() # 이 비용 함수는 소프트맥스 함수를 포함하고 있음.
optimizer = optim.Adam(model.parameters())

losses = []

#training
for epoch in range(100):
  optimizer.zero_grad()
  y_pred = model(X) # forwar 연산
  loss = loss_fn(y_pred, Y)
  loss.backward()
  optimizer.step()

  if epoch % 10 == 0:
    print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, 100, loss.item()
        ))

  losses.append(loss.item())

plt.plot(losses)