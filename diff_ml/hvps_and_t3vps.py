import jax
import jax.numpy as jnp
import equinox as eqx



def hvp(f, x, v):
    return jax.jvp(lambda x_: eqx.filter_grad(f)(x_), (x,), (v,))[1]


# this is a version where we explicitly add a list of boolean values, indicating whether we should compute the hvp for that direction
# TODO isnt this the same as just setting the undesired directions to 0? Then we could just use the normal hvp_batch function
def hvp_cond(f, x, v, eval_hvp):                                   #jnp.zeros(shape=(x.shape[-1],))
    return jax.lax.cond(eval_hvp, lambda _: hvp(f, x, v), lambda _: jnp.zeros_like(v), None)



# from bachelier
def hvp_batch(f, inputs, directions):
    """
    Compute Hessian-vector products: H(x_i) @ v_j
    Args:
        f: scalar-valued function f: R^n -> R
        inputs: [num_inputs, input_dim]
        directions: [num_directions, input_dim]
    Returns:
        hvps: [num_inputs, num_directions, input_dim]
    """
    def hvp_fn(x, v):
        return hvp(f, x, v)
    batched = eqx.filter_vmap(eqx.filter_vmap(hvp_fn, in_axes=(0, None)), in_axes=(None, 0))
    return jnp.transpose(batched(inputs, directions), (1, 0, 2))



def hvp_batch_cond(f, inputs, directions, eval_hvp):
    """
    Compute Hessian-vector products: H(x_i) @ v_j ifff eval_hvp[j] == True
    Args:
        f: scalar-valued function f: R^n -> R
        inputs: [num_inputs, input_dim]
        directions: [num_directions, input_dim]
        eval_hvp: [num_directions] boolean array
            if True, compute H(x_i) @ v_j
            if False, return 0
    Returns:
        hvps: [num_inputs, num_directions, input_dim]
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
        f: scalar-valued function f: R^n -> R
        inputs: [batch_size, input_dim]          (b, d)
        directions: [batch_size, num_dirs, input_dim]  (b, k, d)
    
    Returns:
        hvps: [batch_size, num_dirs, input_dim]  (b, k, d)
    """
    def hvp_fn(x, v):
        return hvp(f, x, v)  # returns H(x) @ v

    def per_sample_hvp(xi, vis):
        return jax.vmap(lambda v: hvp_fn(xi, v))(vis)  # shape (k, d)

    return jax.vmap(per_sample_hvp)(inputs, directions)  # shape (b, k, d)



# f: R^d -> scalar
def tvp(f, x, v, w):
    """Third-derivative contraction with two vectors: T(x)(·, v, w) ∈ R^d."""
    # g(x) = D_w f(x) (scalar)
    def g(x_):
        return jax.jvp(f, (x_,), (w,))[1]

    # h(x) = D_v g(x) = D_v D_w f(x) (scalar)
    def h(x_):
        return jax.jvp(g, (x_,), (v,))[1]

    # ∇ h(x) = T(x)(·, v, w) ∈ R^d
    return eqx.filter_grad(h)(x)



def tvp_batch(f, inputs, v_dirs, w_dirs):
    """
    inputs:     [b, d]
    v_dirs:     [k, d]
    w_dirs:     [k, d]
    returns:    [b, k, k, d] with (i,j) -> T(·, v_i, w_j)
    """
    def tvp_vw(x, v, w):
        return tvp(f, x, v, w)

    # map over inputs (b) and v (k) and w (k)
    batched = eqx.filter_vmap(                        # over v
                eqx.filter_vmap(                      # over w
                    eqx.filter_vmap(tvp_vw, in_axes=(0, None, None)),  # over inputs
                    in_axes=(None, None, 0)),
               in_axes=(None, 0, None))
    # result shape: [k, k, b, d]
    out = batched(inputs, v_dirs, w_dirs)
    return jnp.transpose(out, (2, 0, 1, 3))           # [b, k, k, d]
