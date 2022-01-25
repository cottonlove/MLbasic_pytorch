#import
import numpy as np
import torch
from typing_extensions import Concatenate

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
print(t.sum(dim=1)) # 열을 제거 #tensor([[3.], [7.]])
print(t.sum(dim=-1)) # 열을 제거 #tensor([[3.], [7.]])

#5. Max and Argmax
'''Max: returns maximum value of element
Argmax: returns index of element which has maximum value'''

t = torch.FloatTensor([[1, 2], [3, 4]])
print(t)
print(t.max()) # Returns one value: max tensor(4.)

print(t.max(dim=0)) #remove row Returns two values: max and argmax
#torch.return_types.max(values=tensor([3., 4.]),indices=tensor([1, 1]))

#To get only max or aragmax, use indexing
print('Max: ', t.max(dim=0)[0]) #Max: tensor([3., 4.])
print('Argmax: ', t.max(dim=0)[1]) #Argmax:  tensor([1, 1])

print(t.max(dim=1)) #torch.return_types.max(values=tensor([2., 4.]),indices=tensor([1, 1]))
print(t.max(dim=-1)) #torch.return_types.max(values=tensor([2., 4.]),indices=tensor([1, 1]))

# from typing_extensions import Concatenate
# #import
# import numpy as np
# import torch

'''view(), squeeze(), and unsqueeze() adjust their shape and dimension 
while keeping the number of elements'''

#6.View = Reshape(Numpy)
'''*** Very important ****
Resize(reshape) the tensor while keeping the number of elements'''
t = np.array([[[0, 1, 2],
               [3, 4, 5]],
              [[6, 7, 8],
               [9, 10, 11]]])
ft = torch.FloatTensor(t)

print(ft.shape) #torch.Size([2, 2, 3])

#Change from 3D tensor to 2D tensor
print(ft.view([-1, 3])) # Change ft tensor to a size of (?, 3)
#tensor([[ 0.,  1.,  2.],[ 3.,  4.,  5.],[ 6.,  7.,  8.],[ 9., 10., 11.]])
print(ft.view([-1, 3]).shape) #torch.Size([4, 3])

#Change the shape while maintaining the 3D dimension 
print(ft.view([-1, 1, 3])) #tensor([[[ 0.,  1.,  2.]], [[ 3.,  4.,  5.]], [[ 6.,  7.,  8.]],[[ 9., 10., 11.]]])
print(ft.view([-1, 1, 3]).shape) #torch.Size([4, 1, 3])

#7. Squeeze
'''Remove dimension which is 1'''
ft = torch.FloatTensor([[0], [1], [2]])
print(ft)
print(ft.shape) #torch.Size([3, 1])

print(ft.squeeze()) #remove 2th dimension which is 1. -> tensor([0., 1., 2.])
print(ft.squeeze().shape) #torch.Size([3])

#8. Unsqueeze
'''특정 위치에 1인 차원을 추가한다'''
ft = torch.Tensor([0, 1, 2])
print(ft.shape) #(3,) torch.Size([3])
#첫번째 차원에 1인 차원을 추가
print(ft.unsqueeze(0)) # 인덱스가 0부터 시작하므로 0은 첫번째 차원을 의미
#tensor([[0., 1., 2.]])
print(ft.unsqueeze(0).shape) #torch.Size([1, 3])
#(3,)의 크기를 가졌던 1차원 벡터가 (1, 3)의 2차원 텐서로 변경됨
#view로 구현시 다음과 같음: 2차원으로 바꾸고 싶으면서 첫번째 차원은 1이기를 원한다면 view에서 (1, -1)을 인자로 사용

print(ft.view(1, -1))
print(ft.view(1, -1).shape) #torch.Size([1, 3])

print(ft.unsqueeze(1)) #tensor([[0.],[1.],[2.]])
print(ft.unsqueeze(1).shape) #torch.Size([3, 1])

#인자로 -1추가
#-1은 인덱스 상으로 마지막 차원을 의미=마지막 차원에 1인 차원을 추가
print(ft.unsqueeze(-1)) #tensor([[0.],[1.],[2.]])
print(ft.unsqueeze(-1).shape) #torch.Size([3, 1])

#9. Type Casting
'''Convert data type'''
lt = torch.LongTensor([1, 2, 3, 4])
print(lt) #tensor([1, 2, 3, 4])
#type casting: convert long -> float
print(lt.float()) #tensor([1., 2., 3., 4.])

bt = torch.ByteTensor([True, False, False, True])
print(bt) #tensor([1, 0, 0, 1], dtype=torch.uint8)
#type casting: convert byte -> long/float
print(bt.long()) #tensor([1, 0, 0, 1])
print(bt.float()) #tensor([1., 0., 0., 1.])

#10. Concatenate
'''두 텐서를 연결. torch.cat([ ]). 어느 차원을 늘릴 것인지를 인자로 '''
'''딥 러닝에서는 주로 모델의 입력 또는 중간 연산에서 두 개의 텐서를 연결하는 경우가 많습니다. 
두 텐서를 연결해서 입력으로 사용하는 것은 두 가지의 정보를 모두 사용한다는 의미를 가지고 있습니다.'''
x = torch.FloatTensor([[1, 2], [3, 4]]) #|x| = (2,2)
y = torch.FloatTensor([[5, 6], [7, 8]]) #|y| = (2,2)

print(torch.cat([x, y], dim=0)) #(4,2). tensor([[1., 2.], [3., 4.],[5., 6.],[7., 8.]])
print(torch.cat([x, y], dim=1)) #(2,4). tensor([[1., 2., 5., 6.],[3., 4., 7., 8.]])

#11. Stacking
'''연결(concatenate)을 하는 또 다른 방법
연결을 하는 것보다 스택킹이 더 편리할 때가 있는데, 이는 스택킹이 많은 연산을 포함하고 있기때문'''
x = torch.FloatTensor([1, 4]) #|x|=|y|=|z|=(2,)
y = torch.FloatTensor([2, 5])
z = torch.FloatTensor([3, 6])
print(torch.stack([x, y, z])) #tensor([[1., 4.],[2., 5.],[3., 6.]]) size: (3,2)
#스택킹은 사실 많은 연산을 한 번에 축약하고 있음.위 작업은 아래의 코드와 동일
print(torch.cat([x.unsqueeze(0), y.unsqueeze(0), z.unsqueeze(0)], dim=0))
#x.unsqueeze(0),y.unsqueeze(0),z.unsqueeze(0): (2,) -> (1,2)
#cat(~~~, dim = 0): (1,2) -> (3,2)

print(torch.stack([x, y, z], dim=1)) #두번째 차원이 증가하도록 쌓으라는 의미
#[[1],[4]], [[2],[5]], [[3],[6]] -> tensor([[1., 2., 3.],[4., 5., 6.]])



'''
torch.cat() vs torch.stack()

torch.cat()은 주어진 차원을 기준으로 주어진 텐서들을 붙임(concatenate)
torch.stack()은 새로운 차원으로 주어진 텐서들을 붙입니다.
따라서, (3, 4)의 크기(shape)를 갖는 2개의 텐서 A와 B를 붙이는 경우,
torch.cat([A, B], dim=0)의 결과는 (6, 4)의 크기(shape)를 갖고,
torch.stack([A, B], dim=0)의 결과는 (2, 3, 4)의 크기를 가짐.
'''



#12. One and Zeros
'''0으로 채워진 텐서와 1로 채워진 텐서
동일한 크기(shape)지만 0/1으로만 값이 채워진 텐서를 생성'''
x = torch.FloatTensor([[0, 1, 2], [2, 1, 0]]) #|x| = (2,3)
print(torch.ones_like(x)) #tensor([[1., 1., 1.], [1., 1., 1.]])
print(torch.zeros_like(x)) #tensor([[0., 0., 0.],[0., 0., 0.]])

#13. In-place Operation (덮어쓰기 연산)
'''_붙임. 메모리에 새로 선언하지 않고 기존 tensor에 저장'''
x = torch.FloatTensor([[1, 2], [3, 4]]) 
print(x.mul(2.)) #tensor([[2., 4.],[6., 8.]])
print(x) #tensor([[1., 2.], [3., 4.]])

#In-Place operation
print(x.mul_(2.))#tensor([[2., 4.],[6., 8.]])
print(x)#tensor([[2., 4.],[6., 8.]])