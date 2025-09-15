"""
Autodiff utilities for Hessian-vector products and third-derivative-vector products
"""
import jax
import jax.numpy as jnp
import equinox as eqx



def hvp(f, x, v):
    """
    Compute Hessian-vector product: H(x) @ v
    Args:
        f: scalar-valued function f: R^d -> R
        x: input point x (d)
        v: direction vector  R^d
    Returns:
        hvp: [d]
    """
    return jax.jvp(lambda x_: eqx.filter_grad(f)(x_), (x,), (v,))[1]


# this is a version where we explicitly add a list of boolean values, indicating whether we should compute the hvp for that direction
def hvp_cond(f, x, v, eval_hvp):
    """
    Compute Hessian-vector product H(x) @ v iff eval_hvp == True, else return 0
    """                                   
    return jax.lax.cond(eval_hvp, lambda _: hvp(f, x, v), lambda _: jnp.zeros_like(v), None)



def hvp_batch(f, inputs, directions):
    """
    Compute Hessian-vector products: H(x_i) @ v_j for all points into all directions
    Args:
        f: scalar-valued function f: R^d -> R
        inputs: [b, d]
        directions: [k, d]
    Returns:
        hvps: [b, k, d]
    """
    def hvp_fn(x, v):
        return hvp(f, x, v)
    batched = eqx.filter_vmap(eqx.filter_vmap(hvp_fn, in_axes=(0, None)), in_axes=(None, 0))
    return jnp.transpose(batched(inputs, directions), (1, 0, 2))



def hvp_batch_cond(f, inputs, directions, eval_hvp):
    """
    Compute Hessian-vector products: H(x_i) @ v_j iff eval_hvp[j] == True
    Args:
        f: scalar-valued function f: R^d -> R
        inputs: [b, d]
        directions: [k, d]
        eval_hvp: [k] boolean array
    Returns:
        hvps: [b, k, d]
            where hvps[i, j] = np.zeros if eval_hvp[j] == False
    """
    def hvp_cond_fn(x, v, eval_hvp):
        return hvp_cond(f, x, v, eval_hvp)
    batched = eqx.filter_vmap(eqx.filter_vmap(hvp_cond_fn, in_axes=(0, None, None)), in_axes=(None, 0, 0))
    return jnp.transpose(batched(inputs, directions, eval_hvp), (1, 0, 2))



def hvp_batch_per_input(f, inputs, directions):
    """
    Compute Hessian-vector products H(x_i) @ v_{i,j} for each input x_i and its own set of k directions v_{i,j}.
    Args:
        f: scalar-valued function f: R^d -> R
        inputs: [b, d]
        directions: [b, k, d]
    Returns:
        hvps: [b, k, d]
    """
    def hvp_fn(x, v):
        return hvp(f, x, v)  # returns H(x) @ v

    def per_sample_hvp(xi, vis):
        return jax.vmap(lambda v: hvp_fn(xi, v))(vis)  # shape (k, d)

    return jax.vmap(per_sample_hvp)(inputs, directions)  # shape (b, k, d)



def t3vp(f, x, v, w):
    """
    Third-derivative with two directions
    Args:
        f: scalar-valued function f: R^d -> R
        x: input point x (d)
        v: direction vector  (d)
        w: direction vector  (d)
    Returns:
        tvp: [d]
    """
    # g(x) = D_w f(x) (scalar)
    def g(x_):
        return jax.jvp(f, (x_,), (w,))[1]
    # h(x) = D_v g(x) = D_v D_w f(x) (scalar)
    def h(x_):
        return jax.jvp(g, (x_,), (v,))[1]
    
    # return D h(x) (d)
    return eqx.filter_grad(h)(x)



def t3vp_batch(f, inputs, v_dirs, w_dirs):
    """
    Third derivative with two directions, batched over inputs and directions
    Args:
        f: scalar-valued function f: R^d -> R
        inputs: [b, d]
        v_dirs: [k, d]
        w_dirs: [k, d]
    Returns:
        tvps: [b, k, k, d]
    """
    def t3vp_vw(x, v, w):
        return t3vp(f, x, v, w)

    # map over inputs (b) and v (k) and w (k)
    batched = eqx.filter_vmap(                        # over v
                eqx.filter_vmap(                      # over w
                    eqx.filter_vmap(t3vp_vw, in_axes=(0, None, None)),  # over inputs
                    in_axes=(None, None, 0)),
               in_axes=(None, 0, None))
    # result shape: [k, k, b, d]
    out = batched(inputs, v_dirs, w_dirs)
    return jnp.transpose(out, (2, 0, 1, 3))           # [b, k, k, d]
