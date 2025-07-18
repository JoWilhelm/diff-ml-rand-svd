import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Tuple

from functools import partial



# TODO understand this stuff
# TODO structure better



#def hmp(f, primals):
#    
#    def hvp_(tangents):
#        return hvp(f, (primals,), (tangents, ))
#
#    return eqx.filter_vmap(hvp_)
#
#
#def hvp_args(f, primals, tangents, *f_args):
#    return jax.jvp(lambda x: eqx.filter_grad(f)(x, *f_args), primals, tangents)[1]
#
#def hmp_args(f, primals, *f_args):
#    
#    def hvp_(tangents):
#        return hvp_args(f, (primals,), (tangents, ), *f_args)
#
#    return eqx.filter_vmap(hvp_)
#
#def batch_hmp_args(f, vmapped_args: Tuple = ()):
#
#    def hvp_(primals, tangents, *f_args):
#        return hvp_args(f, (primals, ), (tangents, ), *f_args)
#
#    return eqx.filter_vmap(eqx.filter_vmap(hvp_, in_axes=(0, None, *vmapped_args)), in_axes=(None, 1, *tuple([None for _ in range(len(vmapped_args))])))
#
## jax cannot deal with dynamic slices of arrays
## therefore, we cannot simply slice the principal_components array while using batch_hmp
#
## this is a version where a 0 vector principal component will lead to a different path (namely returning 0) compared to computing a hvp.
#def cond_fn_pca(tangents, *args):
#    # xs, cum_sum = x
#    # return cum_sum > 0.95
#
#    # jax.debug.print("tangents {tangents}", tangents=tangents)
#
#    return jnp.any(tangents[0] > 0.0)  # NOTE: we set the tangents to zero if we do not want to compute its derivative (because principle component is too small)
#
#def hvp_pca(f, primals, tangents):
#    return jax.lax.cond(cond_fn_pca(tangents), lambda _: hvp(f, primals, tangents), lambda _: tangents[0], None)
#
#def batch_hmp_select(f):
#
#    def hvp_(primals, tangents):
#        return hvp_pca(f, (primals,), (tangents,))
#
#    # x = jnp.zeros(shape=(1, 1))
#    # jax.lax.while_loop(cond_fn_pca, fn, (x, 0.0)) # not reverse-mode differentiable!
#    # jnp.piecewise(x, cond_fn_pca, fn, (x, 0.0)) # not reverse-mode differentiable!
#
#    return eqx.filter_vmap(eqx.filter_vmap(hvp_, in_axes=(0, None)), in_axes=(None, 1))
#







#def hvp(f, point, direction):
#    return jax.jvp(lambda x: eqx.filter_grad(f)(x), (point,), (direction,))[1]
#
#
## HVPs in PCA directions
#def get_HVPs(f, primals, tangents):
#
#    def batch_hmp_single_tangent(f, primals):
#            
#        def hvp_(primal, tangent):
#            return hvp(f, primal, tangent)
#
#        return partial(eqx.filter_vmap(hvp_, in_axes=(0, None)), primals)
#
#    # TODO: having input data x coupled is probably not a good idea. We want to reuse the same while_fn for every iteration.
#    #       So make input data a parameter to while_fn as well.
#    single_hvp_fn = batch_hmp_single_tangent(f, primals)
#
#    def while_fn(tup):
#        i, tangents, mtx = tup
#        # jax.debug.print("current x = {x}", x=x)
#        # jax.debug.print("mtx.shape is {mtx}", mtx=mtx.shape)
#        hvps = single_hvp_fn(tangents[i, :].T)
#        # jax.debug.print("test.shape is {ts}", ts=test.shape)
#        mtx = mtx.at[:, i, :].set(hvps)
#        # jax.debug.print("mtx is {mtx}", mtx=mtx)
#
#        return i - 1, tangents, mtx
#
#    # TODO understand if this is correct
#    num_tangents = tangents.shape[0]
#    batch_size, input_dim = primals.shape
#
#    init_state = (
#        num_tangents - 1,  # loop index (starting from last direction)
#        tangents,          # all tangent vectors
#        jnp.zeros((batch_size, num_tangents, input_dim))  # result tensor
#    )
#
#    _, _, mtx = jax.lax.while_loop(lambda tup: tup[0] >= 0, while_fn, init_state)
#    return mtx




#def get_HVPs_conditional():
#    batch_hmp_cond_fn = batch_hmp_cond(MakeScalar(model))
#    hmp_pc_cond = batch_hmp_cond_fn(x, principal_components.T, compute_hvp)
#    hmp_pc_cond = jnp.transpose(hmp_pc_cond, (1, 0, 2))









#def batch_hmp(f):
#
#    def hvp_(primals, tangents):
#        return hvp(f, (primals, ), (tangents, ))
#
#    return eqx.filter_vmap(eqx.filter_vmap(hvp_, in_axes=(0, None)), in_axes=(None, 1))






#def compute_hvp_batch(f, inputs, directions):
#    """
#    Compute HVPs for a scalar-valued function f using nested vmap:
#        H(x_i) @ v_j  for each x_i in primals, each v_j in tangents
#
#    Args:
#        f: scalar-valued function f: R^n -> R
#        primals: [batch_size, input_dim]
#        tangents: [num_directions, input_dim]
#
#    Returns:
#        hvps: [batch_size, num_directions, input_dim]
#    """
#    #def hvp_(x, v):
#    #    return hvp(f=f, point=(x,), direction=(v,))
#
#    hvp_batched = eqx.filter_vmap(
#        eqx.filter_vmap(hvp, in_axes=(0, None)),  # over inputs
#        in_axes=(None, 1)                          # over directions
#    )
#    return jnp.transpose(hvp_batched(inputs, directions), (1, 0, 2))



#def make_hvp_fn(f):
#    def hvp_fn(point, direction):
#        return jax.jvp(lambda x: eqx.filter_grad(f)(x), (point,), (direction,))[1]
#    return hvp_fn
#
#
#def compute_hvp_batch(f, inputs, directions):
#    """
#    Compute Hessian-vector products: H(x_i) @ v_j
#    Args:
#        f: scalar-valued function f: R^n -> R
#        inputs: [batch_size, input_dim]
#        directions: [num_directions, input_dim]
#    Returns:
#        hvps: [batch_size, num_directions, input_dim]
#    """
#    hvp_fn = make_hvp_fn(f)
#
#    # vmap over inputs (x_i) first, then over directions (v_j)
#    batched = eqx.filter_vmap(
#        eqx.filter_vmap(hvp_fn, in_axes=(0, None)),  # x_i
#        in_axes=(None, 0)                            # v_j
#    )
#
#    # result: [num_directions, batch_size, input_dim]
#    hvps = batched(inputs, directions)
#    return jnp.transpose(hvps, (1, 0, 2))  # [batch, directions, input_dim]









# HVP for single point and single direction
def hvp(f, x, v):
    return jax.jvp(lambda x_: eqx.filter_grad(f)(x_), (x,), (v,))[1]

# this is a version where we explicitly add a list of boolean values, indicating whether we should compute the hvp for that direction
# TODO isnt this the same as just setting the undesired directions to 0? Then we could just use the normal hvp_batch function
def hvp_cond(f, x, v, eval_hvp):                                   #jnp.zeros(shape=(x.shape[-1],))
    return jax.lax.cond(eval_hvp, lambda _: hvp(f, x, v), lambda _: jnp.zeros_like(v), None)



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








# TODO put cfd stuff this somewhere else, separate file?
def cfd_fn(f, h, x, *args):
    
    def cfd_(direction):
      #jax.debug.print("in cfd_ direction.shape: {shape}", shape=direction.shape)
      
      xph = x + h * direction 
      xmh = x - h * direction 
      fd_of_f = (f(xph, *args) - f(xmh, *args)) / (2 * h)

      #jax.debug.print("in cfd_ fd_of_f.shape: {fd}", fd=fd_of_f.shape)
      return fd_of_f
    
    return cfd_




def cfd_cond_fn(f, batch_size):
    def cfd_cond_(direction, eval_hvp):
        return jax.lax.cond(eval_hvp, lambda direction: f(direction), lambda _: jnp.zeros(shape=(batch_size, direction.shape[-1])), direction)
        
    return eqx.filter_vmap(cfd_cond_, in_axes=(0, 0))





