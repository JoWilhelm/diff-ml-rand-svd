import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax import vmap
import equinox as eqx

from jaxtyping import Array, PyTree, PRNGKeyArray
from typing import Tuple

from diff_ml.losses.regression import standard_loss_fn, first_order_loss_fn, second_order_loss_fn, third_order_loss_fn
from diff_ml.losses.directions import StreamingHessianSketch
from diff_ml.reference_models.reference_model_class import ReferenceModel
from diff_ml.utils import mse, rmse, MakeScalar
from diff_ml.approx_metrics import approx_metrics, approx_metrics_per_x
from diff_ml.typing import DifferentialData

import time
import optax





class UncertaintyWeighter(eqx.Module):
    """
    learnable uncertainties for up to 4 tasks
    s = log(sigma^2)
    """
    s0: jnp.ndarray = eqx.field(default_factory=lambda: jnp.array(0.003))
    s1: jnp.ndarray = eqx.field(default_factory=lambda: jnp.array(0.003))
    s2: jnp.ndarray = eqx.field(default_factory=lambda: jnp.array(0.003))
    s3: jnp.ndarray = eqx.field(default_factory=lambda: jnp.array(0.003))

    def combine(self, losses: jnp.ndarray, active_mask: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        combine losses weighted by learned uncertainties
        """
        s = jnp.stack([self.s0, self.s1, self.s2, self.s3])  # (4,)
        w = 0.5 * jnp.exp(-s)                                # (4,)

        # total loss over active tasks: sum( w*L + 0.5*s )
        # where the second term prevents s -> -inf
        total = jnp.sum(w * losses * active_mask) + 0.5 * jnp.sum(s * active_mask)  

        # normalized effective weights for display
        w_active = w * active_mask
        denom = jnp.sum(w_active) + 1e-12
        norm_w = w_active / denom
        #norm_w = jnp.where(active_mask > 0, w_active / denom, 0.0)       

        return total, norm_w
    





class WeightedSurrogate(eqx.Module):
    """
    Wrapper class for surrogate MLP and learnable loss balancing weights
    """
    base: eqx.nn.MLP
    uw: UncertaintyWeighter = eqx.field(default_factory=UncertaintyWeighter)

    def __call__(self, *args, **kwargs):
        # normal call to the underlying MLP
        return self.base(*args, **kwargs)








def total_loss_fn(weighted_model: WeightedSurrogate, batch: DifferentialData, batch_key: PRNGKeyArray, ref_model: ReferenceModel, dirs_per_x: Array | None, Svals: Array | None, variant: str, k: int, learnable_loss_weights: bool = True, do_approx_metrics: bool = False):
    """
    combining 0th, 1st, 2nd and 3rd order losses with either equal weights or learnable weights
    """

    model = weighted_model.base
    
    
    L0 = standard_loss_fn(model, batch) 
    L1 = 0.0
    L2 = 0.0
    L3 = 0.0
    iter_data = {}

    if not variant == "value":
        L1 = first_order_loss_fn(model, batch)
        
        if variant not in ["value", "1st"]:
            (L2, iter_data) = second_order_loss_fn(model, batch, batch_key, ref_model, dirs_per_x, Svals, variant, k)
            
            # approximation metrics for 2nd order
            if not variant == "fullHessian" and do_approx_metrics:
                u_H = iter_data["directions"]
                if variant in ("batchSVD", "random", "3rdBatchSVD", "pcady", "streaming"):
                    iter_data["approximation metrics ref"] = approx_metrics(
                                                               fn=ref_model.reference_fn(),
                                                               x=batch.x, 
                                                               U_dirs=u_H
                                                               )
                if variant in ("perXSVD"):
                    iter_data["approximation metrics ref"] = approx_metrics_per_x(
                                                               fn=ref_model.reference_fn(),
                                                               x=batch.x, 
                                                               dirs_per_x=u_H
                                                               )
            
            if variant == "3rdBatchSVD":
                u_H = iter_data["directions"]
                L3  = third_order_loss_fn(model, batch, batch_key, ref_model, u_H, k)
                

    #### ---- combine losses ---- ####

    if not learnable_loss_weights:
        # weighted equally and constant
        if variant == "value":
            total = L0
            iter_data["eff_w_norm"] = [1, 0, 0, 0]
            return total, iter_data
        elif variant == "1st":
            a = 0.5
            b = 0.5
            total = a*L0 + b*L1
            iter_data["eff_w_norm"] = [a, b, 0, 0]
            return total, iter_data
        elif variant == "3rdBatchSVD":
            a = 1/4
            b = 1/4
            c = 1/4
            d = 1/4
            total = a*L0 + b*L1 + c*L2 + d*L3
            iter_data["eff_w_norm"] = [a, b, c, d]
            return total, iter_data
        else:
            a = 1/3
            b = 1/3
            c = 1/3
            total = a*L0 + b*L1 + c*L2
            iter_data["eff_w_norm"] = [a, b, c, 0]
            return total, iter_data
        
    # learnable loss weights
    L1 = L1 if not variant == "value" else 0.0
    L2 = L2 if variant not in ["value", "1st"] else 0.0
    L3 = L3 if variant == "3rdBatchSVD" else 0.0
    loss_vec = jnp.array([L0, L1, L2, L3])

    if variant == "value":
        active_mask = jnp.array([1, 0, 0, 0])
    elif variant == "1st":
        active_mask = jnp.array([1, 1, 0, 0])
    elif variant == "3rdBatchSVD":
        active_mask = jnp.array([1, 1, 1, 1])
    else:
        active_mask = jnp.array([1, 1, 1, 0])

    total, norm_w = weighted_model.uw.combine(loss_vec, active_mask)

    iter_data["eff_w_norm"] = norm_w

    return total, iter_data







def make_train_step(ref_model: ReferenceModel, optim, batch_size: int, variant: str, k: int, learnable_loss_weights: bool = True, do_approx_metrics: bool = False):
    """
    TODO
    """

    @eqx.filter_jit
    def train_step(weighted_model: WeightedSurrogate, sketch: StreamingHessianSketch | None, opt_state: PyTree, batch_key: PRNGKeyArray):
        
        # get new batch of data from reference model
        batch = ref_model.sample(batch_key, batch_size, order=1)

        sketch_dirs = None
        sketch_svals = None
        # update sketch and pass directions for loss
        if variant == "streaming":
            if sketch is None:
                raise ValueError("sketch must be provided for \'streaming\' variant")
            sketch, sketch_dirs, sketch_svals = sketch.update_batch(batch.x)
            
        # total loss and gradients
        (loss_value, iteration_data), grads = eqx.filter_value_and_grad(
            total_loss_fn, has_aux=True
        )(weighted_model, batch, batch_key, ref_model, sketch_dirs, sketch_svals, variant, k, learnable_loss_weights, do_approx_metrics)

        # optimizer step
        updates, opt_state = optim.update(grads, opt_state, weighted_model)
        weighted_model = eqx.apply_updates(weighted_model, updates)
        return weighted_model, opt_state, loss_value, iteration_data, sketch

    return train_step






def train(
    model: eqx.nn.MLP,
    test_data: DifferentialData,
    optim: optax.GradientTransformation,
    n_epochs: int,
    n_batches_per_epoch: int,
    batch_size: int,
    ref_model: ReferenceModel,
    sketch: StreamingHessianSketch | None,
    variant: str,
    k: int,
    learnable_loss_weights: bool = True,
    do_approx_metrics: bool = False,
    do_test_eval: bool = True
) -> Tuple[WeightedSurrogate, list[dict], StreamingHessianSketch | None, float]:
    """
    TODO
    """

    if test_data.dy is None or test_data.ddy is None:
        raise ValueError("\'test_data\' must contain at least first and second order derivatives for evaluation.")

    # wrap model with learnable loss weights    
    weighted_model = WeightedSurrogate(base=model)
    
    # setup training
    train_step = make_train_step(ref_model, optim, batch_size, variant, k, learnable_loss_weights, do_approx_metrics)
    opt_state = optim.init(eqx.filter(weighted_model, eqx.is_array))
    train_loss = jnp.zeros(1)
    n_steps = n_epochs * n_batches_per_epoch
    print(f"Training for {n_epochs} epochs with {n_batches_per_epoch} batches per epoch and batch size {batch_size}.")

    # initialize test errors
    y_error = dy_error = ddy_error = dddy_error = proj_hvp_rmse = jnp.nan

    # training
    duplicate_batch_keys_over_epochs = True
    if not duplicate_batch_keys_over_epochs:
        keys = jrandom.split(ref_model.key_train, n_steps)
    else:
        batch_keys = jrandom.split(ref_model.key_train, n_batches_per_epoch)
        keys = jnp.tile(batch_keys, (n_epochs, 1)).reshape(-1, 2)
    iteration_datas = []
    sum_batch_times = 0
    for i, batch_key in enumerate(keys):
        
        with jax.profiler.StepTraceAnnotation("Train Step", step_num=i):  

            # track execution time per batch / train step
            t0 = time.perf_counter()
            weighted_model, opt_state, train_loss, iteration_data, sketch = train_step(weighted_model, sketch, opt_state, batch_key)
            _ = jax.block_until_ready(train_loss)
            t1 = time.perf_counter()
            if i >= 3:
                sum_batch_times += (t1 - t0)
                

        # evaluate on test data at end of each epoch
        if do_test_eval and i % n_batches_per_epoch == 0:
            epoch_stats = f"Finished epoch {int(i/n_batches_per_epoch)+1} | Train Loss: {train_loss:.5f}"    

                
            # 0th and 1st order errors
            test_pred_ys, test_pred_dys = vmap(jax.value_and_grad(weighted_model))(test_data.x)
            y_error = jnp.sqrt(mse(test_pred_ys, test_data.y))
            dy_error = jnp.sqrt(mse(test_pred_dys, test_data.dy))
                

            # 2nd order error comparing full hessians
            test_pred_ddys = vmap(jax.hessian(MakeScalar(weighted_model)))(test_data.x)
            test_pred_ddys = test_pred_ddys.reshape(test_data.ddy.shape)
            ddy_error = jnp.sqrt(mse(test_pred_ddys, test_data.ddy))
            if variant == "3rdBatchSVD" and test_data.dddy is not None:
                # 3rd order error comparing full third derivative tensors
                test_pred_dddys = vmap(jax.jacfwd(jax.hessian(MakeScalar(weighted_model))))(test_data.x)
                test_pred_dddys = test_pred_dddys.reshape(test_data.dddy.shape)
                dddy_error = jnp.sqrt(mse(test_pred_dddys, test_data.dddy))
            else: dddy_error = .0


            # 2nd order error only in directions used in loss
            if not variant in ["value", "1st", "fullHessian"]:
        
                U = iteration_data["directions"]
                b = test_data.x.shape[0]
                d = ref_model.n_dims
                H_true = test_data.ddy.reshape(b, d, d)
                H_pred = test_pred_ddys.reshape(b, d, d)

                if variant == "batchSVD" or variant == "random" or variant == "3rdbatchSVD" or variant == "streaming":
                    # batch-shared directions: U_norm  (k, d)
                    HU_true  = jnp.einsum('bij,kj->bki', H_true, U)   # (b, k, d)
                    HU_pred  = jnp.einsum('bij,kj->bki', H_pred, U)   # (b, k, d)

                elif variant == "perXSVD":
                    # per-input directions (b, k, d)
                    # normalize all directions
                    U_stack = U.reshape(-1, H_true.shape[-1])  # (b*k, d)
                    eps = 1e-12
                    U_stack = U_stack / (jnp.linalg.norm(U_stack, axis=1, keepdims=True) + eps)
                    # orthonormal basis spanning the same space
                    Q, _ = jnp.linalg.qr(U_stack.T)  
                    U_shared = Q.T    # (m, d) with m <= b*k
                    # project into shared basis
                    HU_true = jnp.einsum('bij,kj->bki', H_true, U_shared) # (b, m, d) 
                    HU_pred = jnp.einsum('bij,kj->bki', H_pred, U_shared) # (b, m, d)
                else:
                    HU_true = jnp.nan
                    HU_pred = jnp.nan
                # RMSE of projected HVPs
                proj_hvp_rmse = rmse(HU_pred, HU_true)

            # end epoch
            epoch_stats += f" | Test Value Loss: {y_error:.5f}"
            print(epoch_stats)
        
        # log per-batch iteration data
        iteration_data["test value loss"] = y_error
        iteration_data["test grad loss"] = dy_error
        iteration_data["test hess loss"] = ddy_error
        iteration_data["test t3 loss"] = dddy_error
        iteration_data["test proj hess loss"] = proj_hvp_rmse 
        iteration_data["train loss"] = train_loss
        iteration_datas.append(iteration_data)
        # end batch

    # end training loop
    avg_time_per_batch = sum_batch_times / (n_steps - 3)
    print(f"Average execution time per batch: {avg_time_per_batch:.5f}s")

    return weighted_model, iteration_datas, sketch, avg_time_per_batch