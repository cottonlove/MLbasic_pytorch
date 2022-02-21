#immport
import torch
import torch.nn.functional as F

torch.manual_seed(1)

## 1. 파이토치로 소프트맥스의 비용함수 구현(low-level)
z = torch.FloatTensor([1, 2, 3])
#z를 소프트맥스 함수의 입력으로 사용, 결과 확인
#dim: A dimension along which Softmax will be computed (so every slice along dim will sum to 1).
hypothesis = F.softmax(z, dim=0)
print(hypothesis)

hypothesis.sum()
#비용함수 직접 구현
z = torch.rand(3, 5, requires_grad=True)
print(z)
hypothesis = F.softmax(z, dim=1)
print(hypothesis)
print(hypothesis.sum(dim = 1))  # 열을 제거. 각 행별로 합을 더함
#print(hypothesis.sum(dim = 0))
y = torch.randint(5, (3,)).long()
'''Returns a tensor filled with random integers generated uniformly between low (inclusive) and high (exclusive).

The shape of the tensor is defined by the variable argument size.
parameters:
- low (int, optional) – Lowest integer to be drawn from the distribution. Default: 0.

- high (int) – One above the highest integer to be drawn from the distribution.

- size (tuple) – a tuple defining the shape of the output tensor.

'''
print(y)
# 모든 원소가 0의 값을 가진 3 × 5 텐서 생성
y_one_hot = torch.zeros_like(hypothesis) 
y_one_hot.scatter_(1, y.unsqueeze(1), 1)
print(y_one_hot)
print(y.unsqueeze(1))
cost = (y_one_hot * -torch.log(hypothesis)).sum(dim=1).mean()
print(cost)

##2. 파이토치로 소프트맥스의 비용함수 구현(high-level)
# Low level
# 첫번째 수식
(y_one_hot * -torch.log(F.softmax(z, dim=1))).sum(dim=1).mean()
# 두번째 수식
(y_one_hot * - F.log_softmax(z, dim=1)).sum(dim=1).mean()
# High level
# 세번째 수식
F.nll_loss(F.log_softmax(z, dim=1), y)
# 네번째 수식
F.cross_entropy(z, y)
#F.cross_entropy는 비용 함수에 소프트맥스 함수까지 포함하고 있음을 기억하고 있어야 구현 시 혼동하지 않습니다
