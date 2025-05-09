import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Tuple


from functools import partial



# TODO understand this stuff
# TODO structure better



def hvp(f, primals, tangents):
    return jax.jvp(lambda x: eqx.filter_grad(f)(x), primals, tangents)[1]

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


# this is a version where we explicitly add a list of boolean values, indicating whether we should compute the hvp or not
def hvp_conditional(f, primals, tangents, eval_hvp):
    # jax.debug.print("primals {x}", x=primals)
    # jax.debug.print("tangents {x}", x=tangents)
    # jax.debug.print("eval_hvp {x}", x=eval_hvp)

    # jax.lax.cond(eval_hvp, lambda _: jax.debug.print("using hvp: {eval}", eval=eval_hvp), lambda _: jax.debug.print("not using hvp: {eval}", eval=~eval_hvp), None)

    # res = hvp(f, primals, tangents)
    # jax.debug.print("evalhvp.shape {res}", res=res.shape)
    return jax.lax.cond(eval_hvp, lambda _: hvp(f, primals, tangents), lambda _: jnp.zeros(shape=(primals[0].shape[-1],)), None)

def batch_hmp_cond(f):

    def hvp_(primals, tangents, eval_hvp):
        return hvp_conditional(f, (primals,), (tangents,), eval_hvp)

    return eqx.filter_vmap(eqx.filter_vmap(hvp_, in_axes=(0, None, None)), in_axes=(None, 1, 0))


def batch_hmp(f):

    def hvp_(primals, tangents):
        return hvp(f, (primals, ), (tangents, ))

    return eqx.filter_vmap(eqx.filter_vmap(hvp_, in_axes=(0, None)), in_axes=(None, 1))




# HVPs in PCA directions
def get_HVPs(f, primals, tangents):

    def batch_hmp_single_tangent(f, primals):
            
        def hvp_(primals, tangents):
            return hvp(f, (primals,), (tangents,))

        return partial(eqx.filter_vmap(hvp_, in_axes=(0, None)), primals)

    # TODO: having input data x coupled is probably not a good idea. We want to reuse the same while_fn for every iteration.
    #       So make input data a parameter to while_fn as well.
    single_hvp_fn = batch_hmp_single_tangent(f, primals)

    def while_fn(tup):
        i, tangents, mtx = tup
        # jax.debug.print("current x = {x}", x=x)
        # jax.debug.print("mtx.shape is {mtx}", mtx=mtx.shape)
        hvps = single_hvp_fn(tangents[i, :].T)
        # jax.debug.print("test.shape is {ts}", ts=test.shape)
        mtx = mtx.at[:, i, :].set(hvps)
        # jax.debug.print("mtx is {mtx}", mtx=mtx)

        return i - 1, tangents, mtx

    # TODO understand if this is correct
    num_tangents = tangents.shape[0]
    batch_size, input_dim = primals.shape

    init_state = (
        num_tangents - 1,  # loop index (starting from last direction)
        tangents,          # all tangent vectors
        jnp.zeros((batch_size, num_tangents, input_dim))  # result tensor
    )

    _, _, mtx = jax.lax.while_loop(lambda tup: tup[0] >= 0, while_fn, init_state)
    return mtx




def get_HVPs_conditional():
    batch_hmp_cond_fn = batch_hmp_cond(MakeScalar(model))
    hmp_pc_cond = batch_hmp_cond_fn(x, principal_components.T, compute_hvp)
    hmp_pc_cond = jnp.transpose(hmp_pc_cond, (1, 0, 2))




# TODO understand how this is used
batch_hmp_fn = batch_hmp(MakeScalar(model))
basis = jnp.eye(x.shape[-1], dtype=x.dtype)
hmp_res = batch_hmp_fn(x, basis)
hmp_res = jnp.transpose(hmp_res, (1, 0, 2))