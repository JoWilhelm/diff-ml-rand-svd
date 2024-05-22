import equinox as eqx
import jax


def hvp(f, primals, tangents):
    return jax.jvp(lambda x: eqx.filter_grad(f)(x), primals, tangents)[1]


def hmp(f, primals):
    def hvp_(tangents):
        return hvp(f, (primals,), (tangents,))

    return eqx.filter_vmap(hvp_)


#
# def batch_hmp(f):
#
#     def hvp_(primals, tangents):
#         return hvp(f, (primals, ), (tangents, ))
#
#     return eqx.filter_vmap(eqx.filter_vmap(hvp_, in_axes=(0, None)), in_axes=(None, 1))
#
# def hvp_args(f, primals, tangents, *f_args):
#     return jax.jvp(lambda x: eqx.filter_grad(f)(x, *f_args), primals, tangents)[1]
#
# def hmp_args(f, primals, *f_args):
#
#     def hvp_(tangents):
#         return hvp_args(f, (primals,), (tangents, ), *f_args)
#
#     return eqx.filter_vmap(hvp_)
#
# def batch_hmp_args(f, vmapped_args: Tuple = ()):
#
#     def hvp_(primals, tangents, *f_args):
#         return hvp_args(f, (primals, ), (tangents, ), *f_args)
#
#     return eqx.filter_vmap(eqx.filter_vmap(hvp_, in_axes=(0, None, *vmapped_args)), in_axes=(None, 1, *tuple([None for _ in range(len(vmapped_args))])))
#
# # jax cannot deal with dynamic slices of arrays
# # therefore, we cannot simply slice the principal_components array while using batch_hmp
#
# # this is a version where a 0 vector principal component will lead to a different path (namely returning 0) compared to computing a hvp.
# def cond_fn_pca(tangents, *args):
#     # xs, cum_sum = x
#     # return cum_sum > 0.95
#
#     # jax.debug.print("tangents {tangents}", tangents=tangents)
#
#     return jnp.any(tangents[0] > 0.0)  # NOTE: we set the tangents to zero if we do not want to compute its derivative (because principle component is too small)
#
# def hvp_pca(f, primals, tangents):
#     return jax.lax.cond(cond_fn_pca(tangents), lambda _: hvp(f, primals, tangents), lambda _: tangents[0], None)
#
# def batch_hmp_select(f):
#
#     def hvp_(primals, tangents):
#         return hvp_pca(f, (primals,), (tangents,))
#
#     # x = jnp.zeros(shape=(1, 1))
#     # jax.lax.while_loop(cond_fn_pca, fn, (x, 0.0)) # not reverse-mode differentiable!
#     # jnp.piecewise(x, cond_fn_pca, fn, (x, 0.0)) # not reverse-mode differentiable!
#
#     return eqx.filter_vmap(eqx.filter_vmap(hvp_, in_axes=(0, None)), in_axes=(None, 1))
#
# # this is a version where we explicitly add a list of boolean values, indicating whether we should compute the hvp or not
# def hvp_conditional(f, primals, tangents, eval_hvp):
#     # jax.debug.print("primals {x}", x=primals)
#     # jax.debug.print("tangents {x}", x=tangents)
#     # jax.debug.print("eval_hvp {x}", x=eval_hvp)
#
#     # jax.lax.cond(eval_hvp, lambda _: jax.debug.print("using hvp: {eval}", eval=eval_hvp), lambda _: jax.debug.print("not using hvp: {eval}", eval=~eval_hvp), None)
#
#     # res = hvp(f, primals, tangents)
#     # jax.debug.print("evalhvp.shape {res}", res=res.shape)
#     return jax.lax.cond(eval_hvp, lambda _: hvp(f, primals, tangents), lambda _: jnp.zeros(shape=(primals[0].shape[-1],)), None)
#
# def batch_hmp_cond(f):
#
#     def hvp_(primals, tangents, eval_hvp):
#         return hvp_conditional(f, (primals,), (tangents,), eval_hvp)
#
#     return eqx.filter_vmap(eqx.filter_vmap(hvp_, in_axes=(0, None, None)), in_axes=(None, 1, 0))
