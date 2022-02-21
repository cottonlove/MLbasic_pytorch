#import
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils import data
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import random

#for data visualization
#데이터 시각화
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# USE_CUDA = torch.cuda.is_available() # GPU를 사용가능하면 True, 아니라면 False를 리턴
# print(USE_CUDA)
# device = torch.device("cuda" if USE_CUDA else "cpu") # GPU 사용 가능하면 사용하고 아니면 CPU 사용
# print("다음 기기로 학습합니다:", device)

# for reproducibility
random.seed(777)
torch.manual_seed(777)
# if device == 'cuda':
#     torch.cuda.manual_seed_all(777)

# hyperparameters
training_epochs = 15
batch_size = 100

# MNIST dataset download
mnist_train = dsets.MNIST(root='MNIST_data/',
                          train=True,
                          transform=transforms.ToTensor(),
                          download=True)

mnist_test = dsets.MNIST(root='MNIST_data/',
                         train=False,
                         transform=transforms.ToTensor(),
                         download=True)

# print(mnist_train)
# print(mnist_test)

# dataset loader
'''
batch 학습 사용하기위해 DataLoader사용
'''
data_loader = DataLoader(dataset=mnist_train,
                                          batch_size=batch_size, # 배치 크기는 100
                                          shuffle=True,
                                          drop_last=True)


print(data_loader)
print(len(data_loader)) #600

# 첫번째 iteration에서 나오는 데이터 확인
images, labels = next(iter(data_loader))
print(images)
print(labels)
print(images.shape, labels.shape) #100개의 samples
#torch.Size([100, 1, 28, 28]) torch.Size([100])

#data visualization(데이터 시각화)
# squeeze() 함수는 차원의 원소가 1인 차원을 없애줍니다.
torch_image = torch.squeeze(images[0])
print(torch_image.shape) #torch.Size([28, 28])
image = toch_image.numpy()
print(image.shape) #(28, 28)
plt.title(labels.numpy()[0])
plt.imshow(image, 'gray')
plt.show()

#model 설계
# MNIST data image of shape 28 * 28 = 784
#input_dim = 28*28=784, output_dim = 10 (0~9 label 갯수)
linear = nn.Linear(784, 10, bias=True) #.to(device) : CUDA사용시

#비용함수와 옵티마이저 정의
criterion = nn.CrossEntropyLoss().to(device) # 내부적으로 소프트맥스 함수를 포함하고 있음.
optimizer = torch.optim.SGD(linear.parameters(), lr=0.1)

#Training
for epoch in range(training_epochs): # 앞서 training_epochs의 값은 15로 지정함.
    avg_cost = 0
    total_batch = len(data_loader)

    for X, Y in data_loader:
        # 배치 크기가 100이므로 아래의 연산에서 X는 (100, 784)의 텐서가 된다.
        X = X.view(-1, 28 * 28) #.to(device): CUDA사용시
        # 레이블은 원-핫 인코딩이 된 상태가 아니라 0 ~ 9의 정수.
        Y = Y #.to(device): CUDA사용시

        optimizer.zero_grad() #꼭 gradient 0으로 초기화해줘야함!
        hypothesis = linear(X)
        cost = criterion(hypothesis, Y)
        cost.backward()
        optimizer.step()

        avg_cost += cost / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

print('Learning finished')

#test_data를 사용해 model 테스트
with torch.no_grad(): # torch.no_grad()를 하면 gradient 계산을 수행하지 않는다.
    X_test = mnist_test.test_data.view(-1, 28 * 28).float() #.to(device): CUDA사용시
    Y_test = mnist_test.test_labels #.to(device): CUDA사용시

    prediction = linear(X_test)
    print(prediction)
    print(torch.argmax(prediction, dim = 1))
    print(Y_test)
    correct_prediction = torch.argmax(prediction, dim = 1) == Y_test
    print(correct_prediction)
    #torch.argmax(prediction, dim = 1): 열을 없애는것. 각 행에서 가장 큰 값을 갖는 index값만 남음.
    #(100,1)형식의 벡터가 남고, 가장 높은 확률은 갖는 index만 나옴. -> 예측된 숫자값
    accuracy = correct_prediction.float().mean()
    print('Accuracy:', accuracy.item())

    # MNIST 테스트 데이터에서 무작위로 하나를 뽑아서 예측을 해본다
    r = random.randint(0, len(mnist_test) - 1)
    X_single_data = mnist_test.test_data[r:r + 1].view(-1, 28 * 28).float().to(device)
    Y_single_data = mnist_test.test_labels[r:r + 1].to(device)

    print('Label: ', Y_single_data.item())
    single_prediction = linear(X_single_data)
    prediction = torch.argmax(single_prediction, 1).item()
    #print(prediction)
    print('Prediction: ', torch.argmax(single_prediction, 1).item())

    plt.title(prediction)
    plt.imshow(mnist_test.test_data[r:r + 1].view(28, 28), cmap='Greys', interpolation='nearest')
    plt.show()
