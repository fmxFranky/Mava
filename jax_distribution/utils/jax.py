import jax
import jax.numpy as jnp
import numpy as np


def ndim_at_least(x, num_dims):
    """Check if the number of dimensions of `x` is at least `num_dims`."""
    if not (isinstance(x, jax.Array) or isinstance(x, np.ndarray)):
        x = jnp.asarray(x)
    return x.ndim >= num_dims


def merge_leading_dims(x, num_dims):
    """Merge leading dimensions.
    Note:
        This implementation is a generic function for merging leading dimensions extracted from Haiku.
        For the original implementation, please refer to the following link:
        (https://github.com/deepmind/dm-haiku/blob/main/haiku/_src/basic.py#L207)
    """
    # Don't merge if there aren't dimensions to merge.
    if not ndim_at_least(x, num_dims):
        return x

    new_shape = (np.prod(x.shape[:num_dims]),) + x.shape[num_dims:]
    return x.reshape(new_shape)
