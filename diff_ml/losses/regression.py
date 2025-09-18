
import jax
from jax import vmap
from jax import random as jrandom
import equinox as eqx

from jaxtyping import Array, Float, PRNGKeyArray

from diff_ml.typing import DifferentialData
from diff_ml.reference_models.reference_model_class import ReferenceModel
from diff_ml.utils import mse, MakeScalar, generate_random_vectors
from diff_ml.losses.directions import get_rand_SVD_directions, get_rand_SVD_directions_per_x, get_3rd_rand_SVD_directions, PCA_of_dydx_directions
from diff_ml.ad import hvp_batch, t3vp_batch, hvp_batch_per_input




def standard_loss_fn(model, batch: DifferentialData):
    """
    0th order loss function
    """
    x = batch.x
    y = batch.y
    y_pred = vmap(model)(x)
    value_loss = mse(y_pred, y)
    return value_loss



def first_order_loss_fn(model, batch: DifferentialData):
    """
    1st order loss function
    """
    dydx_pred = vmap(eqx.filter_grad(MakeScalar(model)))(batch.x)
    assert(dydx_pred.shape == batch.dy.shape)

    grad_loss = mse(dydx_pred, batch.dy)

    return grad_loss
    



@eqx.filter_jit
def second_order_loss_fn(model: eqx.nn.MLP, batch: DifferentialData, key: PRNGKeyArray, ref_model: ReferenceModel, dirs_per_x: Array | None, Svals: Array | None, variant: str, k: int) -> Float:
    """
    2nd order loss function with different ways to get directions to compare HVPs into.
    TODO
    """

    if not variant in ["random", "pcady", "batchSVD", "3rdBatchSVD", "perXSVD", "streaming", "fullHessian"]:
        raise ValueError("variant must be either random, pca, batchSVD, 3rdBatchSVD, perXSVD, streaming or fullHessian")

    iteration_data = {}

    x = batch.x

    
    k = min(k, ref_model.n_dims)  # ensure k does not exceed the number of dimensions

    ref_fn = ref_model.reference_fn()


    mode = "none" # "none", "batch_averaged", "per_input"
    directions = None 


    #### ---- get directions to compare HVPs into ---- ####

    if variant == "fullHessian":
        mode = "none"

    elif variant == "pcady":
        mode = "batch_averaged"
        directions, eval_dir, k_dir = PCA_of_dydx_directions(dydx=batch.dy)
        directions = directions[:k, :] # take top k
        iteration_data["directions"] = directions


    elif variant == "random":
        mode = "batch_averaged"
        key, subkey = jrandom.split(key)
        directions = generate_random_vectors((k, ref_model.n_dims), key=subkey, normalize=True)
        iteration_data["directions"] = directions

    elif variant in ("batchSVD", "3rdBatchSVD"): 
        mode = "batch_averaged"
        directions, eval_dir, k_dir, Svals = get_rand_SVD_directions(
                                        ref_model=ref_model,
                                        f=ref_fn,
                                        x=x,
                                        k=k,
                                        key=key
                                        )
        iteration_data["directions"] = directions
        
    elif variant == "perXSVD":
        mode = "per_input"
        dirs_per_x, eval_dir_batch, k_dir_batch, Svals = get_rand_SVD_directions_per_x(
                                        ref_model=ref_model,
                                        f=ref_fn,
                                        X=x,
                                        k=k, 
                                        key=key
                                        )
        iteration_data["directions"] = dirs_per_x
    
    else: # streaming
        mode = "per_input"
        iteration_data["directions"] = dirs_per_x
        



    #### ---- get 2nd order targets and predictions via HVPs ---- ####
        
    if mode == "per_input":
        
        # targets
        target_hvps = hvp_batch_per_input(
            f=ref_fn,
            inputs=x, 
            directions=dirs_per_x
        )
        #jax.debug.print("targets.shape {shape}", shape=target_hvps.shape)

        # predictions
        pred_hvps = hvp_batch_per_input(
            f=MakeScalar(model),
            inputs=x, 
            directions=dirs_per_x
        )
        #jax.debug.print("preds.shape {shape}", shape=pred_hvps.shape)
        iteration_data["directions"] = dirs_per_x

    
    if mode == "batch_averaged":

        # targets
        target_hvps = hvp_batch(
            f=ref_fn,
            inputs=x, 
            directions=directions
        )
    
        # predictions
        pred_hvps = hvp_batch(
            f=MakeScalar(model),
            inputs=x, 
            directions=directions
        )


    else: # none (full Hessian)
        # true Hessians via jax.hessian for comparison
        target_hvps = vmap(jax.hessian(ref_fn))(x)
        pred_hvps = vmap(jax.hessian(MakeScalar(model)))(x)
        

    #### ---- compute 2nd order loss ---- ####

    assert(target_hvps.shape == pred_hvps.shape)
    
    hess_loss = mse(pred_hvps, target_hvps) 
    #hess_loss = cosine_loss(pred_hvps, target_hvps)
    ## WSE
    #if not variant == "random":
    #    hess_loss = wse(pred_hvps, target_hvps, Svals)
    #else:
    #    hess_loss = mse(pred_hvps, target_hvps)

    return hess_loss, iteration_data






@eqx.filter_jit
def third_order_loss_fn(model: eqx.nn.MLP, batch: DifferentialData, key, ref_model, U_H, k) -> Float:
    """
    3rd order loss function comparing 3rd oder differential Tensor-vector products (T3VPs)
    """
    x = batch.x
    k = min(k, ref_model.n_dims)  # ensure k does not exceed the number of dimensions
    ref_fn = ref_model.reference_fn()

    # get important direction pairs v_i w_i
    dirs_v, dirs_w = get_3rd_rand_SVD_directions(
                                    ref_model=ref_model,
                                    f=ref_fn,
                                    x=x,
                                    U_H=U_H,
                                    k=k
                                    )

    # T3VPs targets
    target_t3vps = t3vp_batch(
        f=ref_fn,
        inputs=x, 
        v_dirs=dirs_v,
        w_dirs=dirs_w
    )

    # T3VPs predictions
    pred_t3vps = t3vp_batch(
        f=MakeScalar(model),
        inputs=x, 
        v_dirs=dirs_v,
        w_dirs=dirs_w
    )

    # loss
    t3_loss = mse(pred_t3vps, target_t3vps)

    return t3_loss




