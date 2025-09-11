import jax
import jax.numpy as jnp
import equinox as eqx


def normalize_vectors(vectors):
    return vectors / jnp.linalg.norm(vectors, axis=1, keepdims=True)

def generate_random_vectors(k, dim, key, normalize=True):
    key, subkey = jax.random.split(key)
    vectors = jax.random.normal(subkey, shape=(k, dim))
    if normalize:
        vectors = normalize_vectors(vectors)
    return vectors


def mse(y_pred, y_true):
    return jnp.mean((y_pred - y_true)**2)

def rmse(y_pred, y_true):
    return jnp.sqrt(mse(y_pred, y_true))

def cosine_loss(p, t, eps=1e-8):
    p_n = p / (jnp.linalg.norm(p, axis=-1, keepdims=True) + eps)
    t_n = t / (jnp.linalg.norm(t, axis=-1, keepdims=True) + eps)
    return jnp.mean(1.0 - jnp.sum(p_n * t_n, axis=-1))


def wse(y_pred, y_true, w):
    """
    Weighted squared error over (b, k, d).
    
    Args:
        y_pred, y_true: arrays of shape (b, k, d)
        w: array of shape (k,) or (b, k), weights sum to 1
    Returns:
        scalar weighted squared error
    """
    diff2 = (y_pred - y_true) ** 2       # (b, k, d)
    per_dir = jnp.mean(diff2, axis=-1)   # (b, k), average over d

    if w.ndim == 1:  # shape (k,)
        # broadcast to (b, k)
        w = jnp.broadcast_to(w, per_dir.shape)

    # now w is (b, k)
    weighted = jnp.sum(w * per_dir, axis=-1)  # (b,)
    return jnp.mean(weighted)                 # scalar




class Range(eqx.Module):
    minval: float = 0.0
    maxval: float = 1.0


class MakeScalar(eqx.Module):
    model: eqx.Module

    def __call__(self, *args, **kwargs):
        out = self.model(*args, **kwargs)
        return jnp.reshape(out, ())
    




#def normalize_vectors(vectors, axis):
#    return vectors / jnp.linalg.norm(vectors, axis=axis, keepdims=True)


def safe_normalize_vectors(vectors, axis, eps=1e-12):
    n = jnp.linalg.norm(vectors, axis=axis, keepdims=True)
    return vectors / (n + eps)

#def generate_random_vectors(k, dim, key, normalize=True):
#    key, subkey = jax.random.split(key)
#    vectors = jax.random.normal(subkey, shape=(k, dim))
#    if normalize:
#        vectors = normalize_vectors(vectors)
#    return vectors



def generate_random_vectors(shape, key, normalize):
   vectors = jax.random.normal(key, shape=shape)
   if normalize:
       vectors = vectors.reshape(vectors.shape[0], -1)
       vectors = safe_normalize_vectors(vectors, axis=-1)
       vectors = vectors.reshape(shape)
   return vectors