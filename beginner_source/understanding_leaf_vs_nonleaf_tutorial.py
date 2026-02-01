"""
requires_grad, retain_grad, 리프(Leaf), 그리고 논-리프(Non-leaf) 텐서(Tensor)들에 대한 이해
====================================================================

**저자:** `Justin Silver <https://github.com/j-silv>`__
**번역:** `최도윤 <https://github.com/justjs4evr>`__

본 튜토리얼은 하나의 예제를 통해 ``requires_grad``,
``retain_grad``, 리프, 그리고 논-리프 텐서들의 세부 사항 및 차이를 설명합니다.

시작하기 전에, `텐서와 텐서 조작법 <https://docs.tutorials.pytorch.kr/beginner/basics/tensorqs_tutorial.html>`_에 대해 이해하고 있는지 확인해 주세요. 
`Autograd의 작동 원리 <https://docs.tutorials.pytorch.kr/beginner/basics/autogradqs_tutorial.html>`_에 대한 기초 지식도 도움이 됩니다.

"""


######################################################################
# 설정(Setup)
# -----
#
# 먼저, `PyTorch가 설치 되어 있는지
# <https://pytorch.org/get-started/locally/>`__ 확인하고,
# 필요한 라이브러리들을 불러옵니다.
#

import torch
import torch.nn.functional as F


######################################################################
# 다음으로, 변화도(Gradient)에 집중하기 위해 간단한 네트워크를 구현합시다.
# Affine 계층과 ReLU 활성화 함수를 거쳐 예측값과 라벨 텐서들
# 사이의 MSE 손실을 구하는 구조입니다.
#
# .. math::
#
#    \mathbf{y}_{\text{pred}} = \text{ReLU}(\mathbf{x} \mathbf{W} + \mathbf{b})
#
# .. math::
#
#    L = \text{MSE}(\mathbf{y}_{\text{pred}}, \mathbf{y})
#
# 참고로, 매개변수(``W`` 그리고 ``b``) 텐서들과 관련된 연산을
# PyTorch가 추적하기 위해선 ``requires_grad=True`` 가 필수입니다.
# `섹션 <#requires-grad>`__에서 이것에 대해 더 다룰 예정입니다.
#

# 텐서 설정하기
x = torch.ones(1, 3)                      # (1, 3) 모양의 입력
W = torch.ones(3, 2, requires_grad=True)  # (3, 2) 모양의 가중치
b = torch.ones(1, 2, requires_grad=True)  # (1, 2) 모양의 편향
y = torch.ones(1, 2)                      # (1, 2) 모양의 출력

# 순전파
z = (x @ W) + b                           # (1, 2) 모양의 활성화 전 단계(pre-activation)
y_pred = F.relu(z)                        # (1, 2) 모양의 활성화 단계
loss = F.mse_loss(y_pred, y)              # 스칼라 손실값


######################################################################
# 리프 vs. 논-리프 텐서들
# -------------------------
#
# 순전파를 수행한 후, PyTorch의 autograd는 아래와 같은 `동적
# 연산
# 그래프 <https://docs.tutorials.pytorch.kr/beginner/blitz/autograd_tutorial.html#computational-graph>`__
# 를 형성했습니다. 이것은 `유향 비순환 그래프(Directed Acyclic Graph, DAG) 
# <https://en.wikipedia.org/wiki/Directed_acyclic_graph>`__인데,
# 입력 텐서 (리프 노드)들, 해당 텐서들에 대한 모든 후속 연산,
# 그리고 중간/출력 텐서 (논-리프 노드)들의 기록을 유지합니다.
# 이 그래프는 미적분의 `연쇄 법칙 <https://en.wikipedia.org/wiki/Chain_rule>`__을
# 적용해 그래프의 루트 (출력)부터 리프 (입력)까지
# 각 텐서의 변화도를 계산하는 데에 사용됩니다:
#
# .. math::
#
#    \mathbf{y} = \mathbf{f}_k\bigl(\mathbf{f}_{k-1}(\dots \mathbf{f}_1(\mathbf{x}) \dots)\bigr)
#
# .. math::
#
#    \frac{\partial \mathbf{y}}{\partial \mathbf{x}} =
#    \frac{\partial \mathbf{f}_k}{\partial \mathbf{f}_{k-1}} \cdot
#    \frac{\partial \mathbf{f}_{k-1}}{\partial \mathbf{f}_{k-2}} \cdot
#    \cdots \cdot
#    \frac{\partial \mathbf{f}_1}{\partial \mathbf{x}}
#
# .. figure:: /_static/img/understanding_leaf_vs_nonleaf/comp-graph-1.png
#    :alt: 순전파 이후의 연산 그래프
#
#    순전파 이후의 연산 그래프
#
# PyTorch는 적어도 하나의 입력이 ``requires_grad=True`` 인
# 텐서 연산의 결과가 아닌 노드를 *리프*(예: ``x``, ``W``, ``b``, ``y``)로 간주합니다.
# 그리고 다른 모든 것은 *논-리프*(예: ``z``, ``y_pred``, ``loss``)죠.
# 텐서의 ``is_leaf`` 속성을 조사하면 확인할 수 있습니다.
#

# True가 출력됩니다. 관례에 따라 새 탠서는 리프이기 때문입니다.
print(f"{x.is_leaf=}")

# False가 출력되는데, 이 텐서가 적어도 하나의 입력이 ``requires_grad=True`` 인
# 연산의 결과이기 때문입니다.
print(f"{z.is_leaf=}")


######################################################################
# 리프와 논-리프 사이의 차이는 텐서의 변화도가 
# 역전파 이후 ``grad`` 속성에 저장될 여부를 결정합니다. 
# 곧, 저장되면 `경사 하강법 <https://en.wikipedia.org/wiki/Gradient_descent>`__
# 에 사용할 수 있을 것입니다. `이어지는 섹션 <#retain-grad>`__에서 이것을
# 좀 더 다룰 예정입니다.
#
# 이제 Pytorch가 연산 그래프에서 텐서의 변화도를 어떻게 계산하고 저장하는 지 살펴봅시다.
#


######################################################################
# ``requires_grad``
# -----------------
#
# 변화도 계산을 위해 사용될 수 있는 연산 그래프를 만들기 위해서,
# ``requires_grad=True`` 매개변수를 텐서 생성시 넣어줄
# 필요가 있습니다. 기본값은 ``False`` 이며, 이때 PyTorch는
# 어떤 만들어진 텐서들에 대해서도 변화도를 추적하지 않습니다. 이를 검증하기 위해,
# ``requires_grad``를 설정하지 않고, 순전파와 역전파를 다시 
# 시행하면 다음과 같은 결과가 나옵니다: 
#
# ::
#
#    >>> loss.backward()
#    RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
#
# 저 에러는 ``loss`` 가 변화도를 추적하지 않고 있기 때문에
# autograd가 그 어느 리프 텐서들에도 역전파할 수 없다는 것을 의미합니다.
# 그 속성을 바꾸려면, 텐서에 ``requires_grad_()`` 를 호출할 수 있습니다. (접미사 \_
# 에 주의하세요).
#
# ``is_leaf`` 속성을 위에서 확인했던 것처럼, 어떤 노드가 변화도 계산을
# 필요로 하는지 확인할 수 있습니다.
#

print(f"{x.requires_grad=}") # False: requires_grad는 False가 기본값이기 때문입니다.
print(f"{W.requires_grad=}") # True: 생성자에서 requires_grad=True로 두었기 때문입니다.
print(f"{z.requires_grad=}") # True: 텐서가 논-리프 노드이기 때문입니다.


######################################################################
# 논-리프 텐서가  정의에 따라 ``requires_grad=True`` 를
# 가지고 있음을 기억하는 것이 중요한데, 아니라면 역전파가 불가능하기
# 때문입니다. 텐서가 리프라면, 유저가 특별하게 정한 경우에만
# ``requires_grad=True`` 를 가지고 있겠죠. 한 텐서에 대한 입력 중 적어도 하나가
# 변화도를 요구하면, 그 텐서 또한 변화도를 요구할 것이라고
# 달리 말할 수 있습니다.
#
# 이 규칙엔 두 예외가 존재합니다:
#
# 1. ``nn.Parameter`` 를 가진 모든 ``nn.Module`` 은
#    그 매개변수들에 대해 ``requires_grad=True`` 를 가질 것입니다.
#    (`참고 <https://docs.tutorials.pytorch.kr/beginner/basics/quickstart_tutorial.html#creating-models>`__)
# 2. 컨텍스트 매니저를 사용해 국소적(local)으로 변화도 계산을 비활성화하는 것도 있습니다.
#    (`참고 <https://docs.pytorch.org/docs/stable/notes/autograd.html#locally-disabling-gradient-computation>`__)
#
# 요약하자면, ``requires_grad`` 는 역전파가 작동하기 위해서
# autograd에게 어느 텐서들이 변화도 계산이 필요한지 알려줍니다.
# 이는 실제로 ``grad`` 필드에 값이 채워지는 텐서가 무엇인지와는 다른 개념인데,
# 다음 섹션에서 다룰 주제입니다.
#


######################################################################
# ``retain_grad``
# ---------------
#
# 실제로 최적화(SGD, Adam, 등)를 수행하려면, 역전파를 실행해
# 변화도를 추출할 필요가 있습니다.
#

loss.backward()


######################################################################
# ``backward()`` 를 호출하면``requires_grad=True`` 를 지닌 
# 모든 리프 텐서들의 ``grad`` 필드를 채워집니다. ``grad`` 는 우리가 조사하는 
# 텐서에 대한 손실(loss)의 변화율을 의미합니다. ``backward()`` 를 실행하기 전에는
# 이 속성이 ``None`` 으로 설정되어 있습니다.
#

print(f"{W.grad=}")
print(f"{b.grad=}")


######################################################################
# 네트워크의 다른 텐서들은 어떻게 될까요? 나머지 리프 노드들을
# 확인해 보겠습니다:
#

# ``requires_grad=False`` 이므로 모두 None을 출력합니다.
print(f"{x.grad=}")
print(f"{y.grad=}")


######################################################################
# PyTorch에 변화도 계산을 명시적으로 요청하지 않았기 때문에(``requires_grad=False``),
# 이 텐서들의 변화도는 채워지지 않았습니다.
#
# 이제 중간 단계의 논-리프 노드를 살펴보겠습니다:
#

print(f"{z.grad=}")


######################################################################
# PyTorch returns ``None`` for the gradient and also warns us that a
# non-leaf node’s ``grad`` attribute is being accessed. Although autograd
# has to calculate intermediate gradients for backpropagation to work, it
# assumes you don’t need to access the values afterwards. To change this
# behavior, we can use the ``retain_grad()`` function on a tensor. This
# tells the autograd engine to populate that tensor’s ``grad`` after
# calling ``backward()``.
#

# we have to re-run the forward pass
z = (x @ W) + b
y_pred = F.relu(z)
loss = F.mse_loss(y_pred, y)

# tell PyTorch to store the gradients after backward()
z.retain_grad()
y_pred.retain_grad()
loss.retain_grad()

# have to zero out gradients otherwise they would accumulate
W.grad = None
b.grad = None

# backpropagation
loss.backward()

# print gradients for all tensors that have requires_grad=True
print(f"{W.grad=}")
print(f"{b.grad=}")
print(f"{z.grad=}")
print(f"{y_pred.grad=}")
print(f"{loss.grad=}")


######################################################################
# We get the same result for ``W.grad`` as before. Also note that because
# the loss is scalar, the gradient of the loss with respect to itself is
# simply ``1.0``.
#
# If we look at the state of the computational graph now, we see that the
# ``retains_grad`` attribute has changed for the intermediate tensors. By
# convention, this attribute will print ``False`` for any leaf node, even
# if it requires its gradient.
#
# .. figure:: /_static/img/understanding_leaf_vs_nonleaf/comp-graph-2.png
#    :alt: Computational graph after backward pass
#
#    Computational graph after backward pass
#
# If you call ``retain_grad()`` on a non-leaf node, it results in a no-op.
# If we call ``retain_grad()`` on a node that has ``requires_grad=False``,
# PyTorch actually throws an error, since it can’t store the gradient if
# it is never calculated.
#
# ::
#
#    >>> x.retain_grad()
#    RuntimeError: can't retain_grad on Tensor that has requires_grad=False
#


######################################################################
# Summary table
# -------------
#
# Using ``retain_grad()`` and ``retains_grad`` only make sense for
# non-leaf nodes, since the ``grad`` attribute will already be populated
# for leaf tensors that have ``requires_grad=True``. By default, these
# non-leaf nodes do not retain (store) their gradient after
# backpropagation. We can change that by rerunning the forward pass,
# telling PyTorch to store the gradients, and then performing
# backpropagation.
#
# The following table can be used as a reference which summarizes the
# above discussions. The following scenarios are the only ones that are
# valid for PyTorch tensors.
#
#
#
# +----------------+------------------------+------------------------+---------------------------------------------------+-------------------------------------+
# |  ``is_leaf``   |   ``requires_grad``    |   ``retains_grad``     |  ``require_grad()``                               |   ``retain_grad()``                 |
# +================+========================+========================+===================================================+=====================================+
# | ``True``       | ``False``              | ``False``              | sets ``requires_grad`` to ``True`` or ``False``   | no-op                               |
# +----------------+------------------------+------------------------+---------------------------------------------------+-------------------------------------+
# | ``True``       | ``True``               | ``False``              | sets ``requires_grad`` to ``True`` or ``False``   | no-op                               |
# +----------------+------------------------+------------------------+---------------------------------------------------+-------------------------------------+
# | ``False``      | ``True``               | ``False``              | no-op                                             | sets ``retains_grad`` to ``True``   |
# +----------------+------------------------+------------------------+---------------------------------------------------+-------------------------------------+
# | ``False``      | ``True``               | ``True``               | no-op                                             | no-op                               |
# +----------------+------------------------+------------------------+---------------------------------------------------+-------------------------------------+
#


######################################################################
# Conclusion
# ----------
#
# In this tutorial, we covered when and how PyTorch computes gradients for
# leaf and non-leaf tensors. By using ``retain_grad``, we can access the
# gradients of intermediate tensors within autograd’s computational graph.
#
# If you would like to learn more about how PyTorch’s autograd system
# works, please visit the `references <#references>`__ below. If you have
# any feedback for this tutorial (improvements, typo fixes, etc.) then
# please use the `PyTorch Forums <https://discuss.pytorch.org/>`__ and/or
# the `issue tracker <https://github.com/pytorchkorea/tutorials-kr/issues>`__ to
# reach out.
#


######################################################################
# References
# ----------
#
# -  `A Gentle Introduction to
#    torch.autograd <https://docs.tutorials.pytorch.kr/beginner/blitz/autograd_tutorial.html>`__
# -  `Automatic Differentiation with
#    torch.autograd <https://docs.tutorials.pytorch.kr/beginner/basics/autogradqs_tutorial>`__
# -  `Autograd
#    mechanics <https://docs.pytorch.org/docs/stable/notes/autograd.html>`__
#