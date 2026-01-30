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
# To build the computational graph which can be used for gradient
# calculation, we need to pass in the ``requires_grad=True`` parameter to
# a tensor constructor. By default, the value is ``False``, and thus
# PyTorch does not track gradients on any created tensors. To verify this,
# try not setting ``requires_grad``, re-run the forward pass, and then run
# backpropagation. You will see:
#
# ::
#
#    >>> loss.backward()
#    RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
#
# This error means that autograd can’t backpropagate to any leaf tensors
# because ``loss`` is not tracking gradients. If you need to change the
# property, you can call ``requires_grad_()`` on the tensor (notice the \_
# suffix).
#
# We can sanity check which nodes require gradient calculation, just like
# we did above with the ``is_leaf`` attribute:
#

print(f"{x.requires_grad=}") # prints False because requires_grad=False by default
print(f"{W.requires_grad=}") # prints True because we set requires_grad=True in constructor
print(f"{z.requires_grad=}") # prints True because tensor is a non-leaf node


######################################################################
# It’s useful to remember that a non-leaf tensor has
# ``requires_grad=True`` by definition, since backpropagation would fail
# otherwise. If the tensor is a leaf, then it will only have
# ``requires_grad=True`` if it was specifically set by the user. Another
# way to phrase this is that if at least one of the inputs to a tensor
# requires the gradient, then it will require the gradient as well.
#
# There are two exceptions to this rule:
#
# 1. Any ``nn.Module`` that has ``nn.Parameter`` will have
#    ``requires_grad=True`` for its parameters (see
#    `here <https://docs.tutorials.pytorch.kr/beginner/basics/quickstart_tutorial.html#creating-models>`__)
# 2. Locally disabling gradient computation with context managers (see
#    `here <https://docs.pytorch.org/docs/stable/notes/autograd.html#locally-disabling-gradient-computation>`__)
#
# In summary, ``requires_grad`` tells autograd which tensors need to have
# their gradients calculated for backpropagation to work. This is
# different from which tensors have their ``grad`` field populated, which
# is the topic of the next section.
#


######################################################################
# ``retain_grad``
# ---------------
#
# To actually perform optimization (e.g. SGD, Adam, etc.), we need to run
# the backward pass so that we can extract the gradients.
#

loss.backward()


######################################################################
# Calling ``backward()`` populates the ``grad`` field of all leaf tensors
# which had ``requires_grad=True``. The ``grad`` is the gradient of the
# loss with respect to the tensor we are probing. Before running
# ``backward()``, this attribute is set to ``None``.
#

print(f"{W.grad=}")
print(f"{b.grad=}")


######################################################################
# You might be wondering about the other tensors in our network. Let’s
# check the remaining leaf nodes:
#

# prints all None because requires_grad=False
print(f"{x.grad=}")
print(f"{y.grad=}")


######################################################################
# The gradients for these tensors haven’t been populated because we did
# not explicitly tell PyTorch to calculate their gradient
# (``requires_grad=False``).
#
# Let’s now look at an intermediate non-leaf node:
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