#import
import numpy as np
import torch

#numpy review
#1D Array with Numpy (Vector)
t = np.array([0., 1., 2., 3., 4., 5., 6.])
#t = np.array([0, 1, 2, 3, 4., 5., 6.])
print(t)

print('Rank of t: ', t.ndim)
print('Shape of t: ', t.shape)

print('t[0] t[1] t[-1] = ', t[0], t[1], t[-1]) #0.0 1.0 6.0
print('t[2:5] t[4:-1] = ', t[2:5], t[4:-1]) #[2., 3., 4.] [4., 5.]
print('t[:2] t[3: ] = ', t[:2], t[3:]) #[0., 1.] [3., 4., 5., 6.]

#2D array with numpy (Metrics)
t = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12.]])
print(t)
print('Rank of t: ', t.ndim) #2차원
print('Shape of t: ', t.shape) #(4,3)

#Pytorch Tensor: similar w/ numpy 
#1D array with pytorch
t = torch.FloatTensor([0., 1., 2., 3., 4., 5., 6.])
print(t)

print(t.dim()) #rank(dimension)) 1
print(t.shape) #shape torch.Size([7])
print(t.size()) #shape torch.Size([7])

print('t[0] t[1] t[-1] = ', t[0], t[1], t[-1]) #tensor(0.) tensor(1.) tensor(6.)
print('t[2:5] t[4:-1] = ', t[2:5], t[4:-1]) #tensor([2., 3., 4.]) tensor([4., 5.])
print('t[:2] t[3: ] = ', t[:2], t[3:]) #tensor([0., 1.]) tensor([3., 4., 5., 6.])

#2D array with pytorch
t = torch.FloatTensor([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12.]])
print(t)

print(t.dim()) #rank(dimension)) 2
print(t.shape) #shape torch.Size([4,3])
print(t.size()) #shape torch.Size([4,3])
print(t[:, 1]) # 첫번째 차원을 전체 선택한 상황에서 두번째 차원의 첫번째 것만 가져온다.
#tensor([2., 3., 8., 11.])
print(t[:, 1].size()) #torch.Size([4]) #a vector of 4 elements
print(t[:, : -1]) # 첫번째 차원을 전체 선택한 상황에서 두번째 차원에서는 맨 마지막에서 첫번째를 제외하고 다 가져온다.
#tensor([[ 1.,  2.], [ 4.,  5.], [ 7.,  8.], [10., 11.]])
print(t[0:4:2, :2] ) #첫번째 차원에서 0번째부터 3번째까지 중 2개씩 이동하면서 가져오기. 마지막 차원은 0-1번째까지 가져오기
#tensor([[1., 2.], [7., 8.]])