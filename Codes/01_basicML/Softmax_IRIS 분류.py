#KNN 붓꽃 (IRIS)의 품종 분류하기 모델
'''
미해결: MNIST처럼 batch 학습 시키고 싶으면 dataload어떻게 해야하지?!?
'''

from sklearn.datasets import load_iris
#import
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import random
import pandas as pd
import numpy

# for reproducibility
random.seed(777)
torch.manual_seed(777)

# hyperparameters
training_epochs = 15
#batch_size = 10

#load iris data
#dataset 로드하기
iris_dataset = load_iris()
# print(iris_dataset)
# #dictionary 형태.

#load한 dataset형태 pandas로 확인하기
dataset = pd.DataFrame(data=numpy.c_[iris_dataset['data'], iris_dataset['target']], columns=iris_dataset['feature_names']+['target']).astype('float32')
dataset.head()

#데이터 형식 파악하기
print(iris_dataset.keys())
print(iris_dataset['target'])
print(iris_dataset['feature_names'])
print(iris_dataset['target_names']) #['setosa' 'versicolor' 'virginica'] -> 3개의 labels
print(iris_dataset['data'].shape) #(150, 4) -> #samples = 150, #features = 4

#One-hot Encoding하는 방법 확인
from sklearn.preprocessing import OneHotEncoder
# from sklearn.preprocessing import LabelEncoder
# label_ids=label_encoder.fit_transform(labels) -> 여기선 이미 target이 숫자로 나와서 안해도 돔.

onehot_encoder=OneHotEncoder(sparse=False)
label_ids = iris_dataset['target']
print(label_ids)
reshaped=label_ids.reshape(len(label_ids), 1)
onehot=onehot_encoder.fit_transform(reshaped)
print(onehot)

#Train, Test dataset나누기
'''
150개의 붓꽃 데이터를 두 그룹으로 나눈다.
scikit-learn은 데이터셋을 섞어서 나눠주는 train_test_split함수를 제공한다.

이 함수는 전체 행중 75%를 훈련세트로 뽑고 나머지 25%는 테스트세트가 된다.(test_size 지정안해주면 default임)
'''
from sklearn.model_selection import train_test_split
X_train, X_test, y_train,y_test = train_test_split(iris_dataset['data'],iris_dataset['target'],test_size = 0.25,random_state=0)


print("shape of X_train is {}".format(X_train.shape))
print("shape of y_train is {}".format(y_train.shape))
print("shape of X_test is {}".format(X_test.shape))
print("shape of y_test is {}".format(y_test.shape))

#One-hot Encoding: y_train, y_test 
reshaped=y_train.reshape(len(y_train), 1)
one_hot_ytrain=onehot_encoder.fit_transform(reshaped)
reshaped=y_test.reshape(len(y_test), 1)
one_hot_ytest = onehot_encoder.fit_transform(reshaped)
print(one_hot_ytrain)
print(one_hot_ytest)
print(one_hot_ytrain.shape)
print(one_hot_ytest.shape)

#모델 구현 및 학습
linear = nn.Linear(4, 3, bias=True)

# 비용 함수와 옵티마이저 정의
criterion = nn.CrossEntropyLoss() # 내부적으로 소프트맥스 함수를 포함하고 있음.
optimizer = torch.optim.SGD(linear.parameters(), lr=0.1)

#Training
for epoch in range(1000): # 앞서 training_epochs의 값은 15로 지정함.
    
    optimizer.zero_grad()
    hypothesis = linear(torch.FloatTensor(X_train)) 
    #***여기서 X_train은 tensor가 아닌 numpy array라서 Tensor로 바꿔주기***
    cost = criterion(hypothesis, torch.FloatTensor(one_hot_ytrain))
    cost.backward()
    optimizer.step()
    if epoch % 100 == 0:
      print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(cost))

print('Learning finished')

#Test data로 모델 테스트하기
with torch.no_grad(): # torch.no_grad()를 하면 gradient 계산을 수행하지 않는다.
    torchX_test = torch.FloatTensor(X_test)
    torchY_test = torch.FloatTensor(one_hot_ytest)

    prediction = linear(torchX_test)
    print(prediction)
    print(torch.argmax(prediction, dim = 1))
    #print("y_test is {}".format(y_test))
    correct_prediction = torch.argmax(prediction, dim = 1) == torch.FloatTensor(y_test)
    #tensor형식이 아니면 오류난다!!!
    
    print(correct_prediction)
    #torch.argmax(prediction, dim = 1): 열을 없애는것. 각 행에서 가장 큰 값을 갖는 index값만 남음.
    #(38,1)형식의 벡터가 남고, 가장 높은 확률은 갖는 index만 나옴. -> 예측된 숫자값
    accuracy = correct_prediction.float().mean()
    print('Accuracy:', accuracy.item())

    #테스트 데이터에서 무작위로 하나를 뽑아서 예측을 해본다
    r = random.randint(0, len(X_test) - 1)
    X_single_data = X_test[r:r + 1]
    Y_single_data = y_test[r:r + 1]

    print('Label: ', Y_single_data.item())
    single_prediction = linear(torch.FloatTensor(X_single_data))
    prediction = torch.argmax(single_prediction, 1).item()
    #print(prediction)
    print('Prediction: ', torch.argmax(single_prediction, 1).item())
