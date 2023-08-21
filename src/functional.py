from typing import Optional

from torch import Tensor, exp

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from torch.types import _dtype as DType
else:
    # The JIT doesn't understand Union, nor torch.dtype here
    DType = int


def softmax_n(input: Tensor, n: Optional[float] = None, dim: Optional[int] = None, dtype: Optional[DType] = None) -> Tensor:
    """
    $\text(softmax)_n(x_i) = exp(x_i) / (n + \sum_j exp(x_j))$

    Note: softmax_n, with fixed input, is _not_ shift-symmetric when n != 0, and we must account for this.
    Normally when computing a softmax, the maxes are subtracted from the inputs for numeric stability.
    """
    if n is None:
        n = 0.
    shift = input.max(dim=dim, keepdim=True).values.detach()
    numerator = exp(input - shift)
    output = numerator / (n * exp(-shift) + numerator.sum(dim=dim, keepdim=True))
    return output if dtype is None else output.type(dtype=dtype)
