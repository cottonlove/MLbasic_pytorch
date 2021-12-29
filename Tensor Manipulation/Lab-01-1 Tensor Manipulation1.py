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

#2D array with numpy (Matrics)
t = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12.]])
print(t)
print('Rank of t: ', t.ndim) #2차원
print('Shape of t: ', t.shape) #(4,3)

#PyTorch Tensor: similar w/ numpy 
#1D array with PyTorch
t = torch.FloatTensor([0., 1., 2., 3., 4., 5., 6.])
print(t)

print(t.dim()) #rank(dimension)) 1
print(t.shape) #shape torch.Size([7])
print(t.size()) #shape torch.Size([7])

print('t[0] t[1] t[-1] = ', t[0], t[1], t[-1]) #tensor(0.) tensor(1.) tensor(6.)
print('t[2:5] t[4:-1] = ', t[2:5], t[4:-1]) #tensor([2., 3., 4.]) tensor([4., 5.])
print('t[:2] t[3: ] = ', t[:2], t[3:]) #tensor([0., 1.]) tensor([3., 4., 5., 6.])

#2D array with PyTorch
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

#1. Broadcasting(automatic -> be careful)

#same shape
m1 = torch.FloatTensor([[3,3]]) #|m1| = (1,2)
m2 = torch.FloatTensor([[2,2]]) #|m2| = (1,2)
print(m1+m2) #tensor([[5.,5.]])

#vector + scalar
m1 = torch.FloatTensor([[1,2]]) #|m1| = (1,2)
m2 = torch.FloatTensor([3]) #3 -> [[3,3]]  |m2| = (1,) -> (1,2)
print(m1+m2) #tensor([[4., 5.]]) <- [[1,2]] + [[3,3]]

#1*2 vector + 2*1 vector
m1 = torch.FloatTensor([[1,2]]) #|m1| = (1,2) -> (2,2)
#print(m1)
m2 = torch.FloatTensor([[3], [4]]) #|m2| = (2,1) -> (2,2)
#print(m2)
print(m1 + m2) #tensor([[4., 5.],[5., 6.]]) <- [[1,2],[1,2]] + [[3,3],[4,4]]

#2. Multiplication vs Matrix Multiplication

print('Matmul vs Mul')

#Matmul: 행렬곱셈 <- matmul()
m1 = torch.FloatTensor([[1, 2], [3, 4]])
m2 = torch.FloatTensor([[1], [2]])
print('Shape of Matrix 1: ', m1.shape) #torch.Size([2, 2] 2 x 2
print('Shape of Matrix 2: ', m2.shape) #torch.Size([2, 1] 2 x 1
print(m1.matmul(m2)) #tensor([[ 5.],[11.]]) size:2 x 1

#Mul(element-wise) <- * or mul()
m1 = torch.FloatTensor([[1, 2], [3, 4]])
m2 = torch.FloatTensor([[1], [2]])
print('Shape of Matrix 1: ', m1.shape) #torch.Size([2, 2] 2 x 2
print('Shape of Matrix 2: ', m2.shape) #torch.Size([2, 1] 2 x 1 -> 2x2
print(m1 * m2) #m2=[[1], [2]] -> [[1,1],[2,2]] (broadcasting)
print(m1.mul(m2)) #tensor([[1., 2.],[6., 8.]])

#3. Mean <- .mean()

t = torch.FloatTensor([1,2])
print(t.mean()) #tensor(1.5000)

t = torch.FloatTensor([[1, 2], [3, 4]])
print(t)
print(t.mean()) #(1+2+3+4)/4 = tensor(2.5000)

#giving dimension as a parameter -> removing such dim
print(t.mean(dim=0)) #remove row so column left (2,2)->(1,2)=(2,):vector
#mean of 1,3 mean of 2,4 -> [2., 3.]

print(t.mean(dim=1)) #remove column so row left (2,2) -> (2,1): 결국 1차원 -> (1,2)로표현
#mean of 1,2 mean of 3,4 -> tensor([1.5000, 3.5000])

print((t.mean(dim=-1))) #remove last dimension so remove column

#4. Sum: the meaning of parameter is same w/ Mean

t = torch.FloatTensor([[1, 2], [3, 4]])
print(t)
print(t.sum()) # 단순히 원소 전체의 덧셈을 수행 #tensor(10.)
print(t.sum(dim=0)) # 행을 제거 #tensor([4., 6.])
print(t.sum(dim=1)) # 열을 제거 #tensor([3., 7.])
print(t.sum(dim=-1)) # 열을 제거 #tensor([3., 7.])

#5. Max and Argmax
'''Max: returns maximum value of element
Argmax: returns index of element which has maximum value'''

t = torch.FloatTensor([[1, 2], [3, 4]])
print(t)
print(t.max()) # Returns one value: max

print(t.max(dim=0)) #remove row Returns two values: max and argmax
#torch.return_types.max(values=tensor([3., 4.]),indices=tensor([1, 1]))

#To get only max or aragmax, use indexing
print('Max: ', t.max(dim=0)[0]) #Max: tensor([3., 4.]
print('Argmax: ', t.max(dim=0)[1]) #Argmax:  tensor([1, 1])

print(t.max(dim=1)) #torch.return_types.max(values=tensor([2., 4.]),indices=tensor([1, 1]))
print(t.max(dim=-1)) #torch.return_types.max(values=tensor([2., 4.]),indices=tensor([1, 1]))