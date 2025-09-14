import equinox as eqx
import jax.numpy as jnp
from typing import Tuple
from diff_ml.typing import DifferentialData

import jax
from jax import vmap
import jax.numpy as jnp
from jax import random as jrandom
import equinox as eqx

import optax


from typing_extensions import TypeAlias
from jaxtyping import Array, Float, PyTree


import jax.numpy as jnp
import jax
import jax.numpy as jnp
import jax.random as jrandom


from diff_ml.losses.regression import standard_loss_fn, first_order_loss_fn, second_order_loss_fn, third_order_loss_fn
from diff_ml.losses.directions import StreamingHessianSketch
from diff_ml.reference_models.reference_model_class import ReferenceModel

from diff_ml.utils import mse, rmse, MakeScalar

from diff_ml.approx_metrics import approx_metrics, approx_metrics_per_x


print(jax.devices())



class UncertaintyWeighter(eqx.Module):
    """Learnable homoscedastic uncertainties for up to 4 tasks (0..3)."""
    # s = log(sigma^2)
    s0: jnp.ndarray = eqx.field(default_factory=lambda: jnp.array(0.003))
    s1: jnp.ndarray = eqx.field(default_factory=lambda: jnp.array(0.003))
    s2: jnp.ndarray = eqx.field(default_factory=lambda: jnp.array(0.003))
    s3: jnp.ndarray = eqx.field(default_factory=lambda: jnp.array(0.003))

    def combine(
        self,
        losses: jnp.ndarray,         # shape (4,)  e.g. [L0, L1, L2, L3]
        active_mask: jnp.ndarray,    # shape (4,)  boolean or {0,1} for which losses are active
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Returns:
          total_loss: scalar  (sum over active tasks of 0.5*exp(-s)*L + 0.5*s)
          norm_w:     shape (4,) normalized effective weights for display, zeros on inactive
        """
        # stack parameters -> shape (4,)
        s = jnp.stack([self.s0, self.s1, self.s2, self.s3])              # (4,)
        w = 0.5 * jnp.exp(-s)                                            # (4,)

        # Ensure mask is float {0,1}
        m = active_mask.astype(losses.dtype)                              # (4,)

        # Total loss over active tasks: sum( w*L + 0.5*s ) on active entries
        total = jnp.sum(w * losses * m) + 0.5 * jnp.sum(s * m)           # scalar

        # Normalized display weights over active tasks only
        w_active = w * m
        denom = jnp.sum(w_active) + 1e-12
        norm_w = jnp.where(m > 0, w_active / denom, 0.0)                 # (4,)

        return total, norm_w


class WeightedSurrogate(eqx.Module):
    base: eqx.Module
    uw: UncertaintyWeighter = eqx.field(default_factory=UncertaintyWeighter)

    def __call__(self, *args, **kwargs):
        return self.base(*args, **kwargs)





def total_loss_fn(weighted_model: WeightedSurrogate, batch: DifferentialData, batch_key, ref_model, dirs, dirs_per_x, Svals, variant: str, k: int, learnable_loss_weights: bool = True):
    
    base = weighted_model.base
    uw   = weighted_model.uw

    # Your existing component losses (all scalars)
    L0 = standard_loss_fn(base, batch)                    # 0th order
    
    
    L1, iter_data = 0.0, {}
    L2 = 0.0
    L3 = 0.0

    if not variant == "value":
        L1 = first_order_loss_fn(base, batch)                 # 1st order
        
        
        if variant not in ["value", "1st"]:
            # second_order_loss_fn returns (loss, iter_data)
            (L2, iter_data) = second_order_loss_fn(base, batch, batch_key, ref_model, dirs, dirs_per_x, Svals, variant, k)
            



            if not variant == "fullHessian":
                # Do approximation metric here with returned directions
                u_H = iter_data.get("directions", None)
    
                x = batch.x
                x_raw_flat = x.reshape(x.shape[0], ref_model.n_dims)
                
                if variant == "batchSVD" or variant == "random" or variant == "3rdBatchSVD":
                    approximation_metrics_ref = approx_metrics(
                                                               fn=ref_model.reference_fn(),
                                                               x=x_raw_flat, 
                                                               U_dirs=u_H
                                                               )
                if variant == "perXSVD" or variant == "streaming":
                    approximation_metrics_ref = approx_metrics_per_x(
                                                               fn=ref_model.reference_fn(),
                                                               x=x_raw_flat, 
                                                               dirs_per_x=u_H
                                                               )
    
    
                iter_data["approximation metrics ref"] = approximation_metrics_ref








            # Optional third-order branch
            if variant == "3rdBatchSVD":
                # You already store U_H in iter_data in your current code
                u_H = iter_data.get("directions", None)
                L3  = third_order_loss_fn(base, batch, batch_key, ref_model, u_H, k)
                


    if not learnable_loss_weights:
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
        

    # Select which losses are active for the chosen variant
    # Order: [L0, L1, L2, L3]
    if variant == "value":
        active = [0]
    elif variant == "1st":
        active = [0, 1]
    elif variant == "3rdBatchSVD":
        active = [0, 1, 2, 3]
    else:
        active = [0, 1, 2]

    L1 = L1 if not variant == "value" else 0.0
    L2 = L2 if variant not in ["value", "1st"] else 0.0
    L3 = L3 if variant == "3rdBatchSVD" else 0.0
    loss_vec = jnp.array([L0, L1, L2, L3])

    if variant == "value":
        mask = jnp.array([1, 0, 0, 0], dtype=loss_vec.dtype)
    elif variant == "1st":
        mask = jnp.array([1, 1, 0, 0], dtype=loss_vec.dtype)
    elif variant == "3rdBatchSVD":
        mask = jnp.array([1, 1, 1, 1], dtype=loss_vec.dtype)
    else:
        mask = jnp.array([1, 1, 1, 0], dtype=loss_vec.dtype)

    total, norm_w = weighted_model.uw.combine(loss_vec, mask)

    # If you want to log the normalized weights per iteration:
    iter_data["eff_w_norm"] = norm_w

    return total, iter_data





import time

def make_train_step(ref_model, optim, batch_size, variant, k, learnable_loss_weights: bool = True):

    @eqx.filter_jit
    def train_step(weighted_model: WeightedSurrogate, sketch: PyTree, opt_state: PyTree, batch_key):
        batch = ref_model.sample(batch_key, batch_size)

        # (your streaming sketching logic unchanged) -> dirs, dirs_per_x, Svals
        dirs = dirs_per_x = Svals = None
        if sketch and variant == "streaming":
            x = batch["x"]
            sketch, refinement_directions, Svals = sketch.update_batch(x)
            dirs = refinement_directions.mean(axis=0)
            dirs_per_x = refinement_directions

        # Single total loss with learnable weights:
        (loss_value, iteration_data), grads = eqx.filter_value_and_grad(
            total_loss_fn, has_aux=True
        )(weighted_model, batch, batch_key, ref_model, dirs, dirs_per_x, Svals, variant, k, learnable_loss_weights)

        updates, opt_state = optim.update(grads, opt_state, weighted_model)
        weighted_model = eqx.apply_updates(weighted_model, updates)
        return weighted_model, opt_state, loss_value, iteration_data, sketch

    return train_step


def train(
    model: PyTree,
    test_data: DifferentialData,
    optim: optax.GradientTransformation,
    n_epochs: int,
    n_batches_per_epoch: int,
    batch_size: int,
    ref_model: ReferenceModel,
    sketch: StreamingHessianSketch,
    variant: str,
    k: int,
    learnable_loss_weights: bool = True,
) -> PyTree:
    

    
    weighted_model = WeightedSurrogate(base=model)


    
    train_step = make_train_step(ref_model, optim, batch_size, variant, k, learnable_loss_weights)
    opt_state = optim.init(eqx.filter(weighted_model, eqx.is_array))
    train_loss = jnp.zeros(1)

    n_steps = n_epochs * n_batches_per_epoch
    print(f"Training for {n_epochs} epochs with {n_batches_per_epoch} batches per epoch and batch size {batch_size}.")
    
    keys = jrandom.split(ref_model.key_train, n_steps)

    #epoch_percent = 0
    iteration_datas = []
    
    sum_batch_times = 0
    
    for i, batch_key in enumerate(keys):
        
        # print(i)
        # print(batch["normalized_initial_states"])
        # print(batch["normalized_payoffs"].shape)
        with jax.profiler.StepTraceAnnotation("Train Step", step_num=i):  

            #weighted_model, opt_state, train_loss, iteration_data, sketch = train_step(weighted_model, sketch, opt_state, batch_key)


            # track execution time per batch 
            t0 = time.perf_counter()
            weighted_model, opt_state, train_loss, iteration_data, sketch = train_step(weighted_model, sketch, opt_state, batch_key)
            _ = jax.block_until_ready(train_loss)
            t1 = time.perf_counter()
            #print(f"Execution time per batch: {t1 - t0:.5f}s")
            sum_batch_times += (t1 -t0)
                

        if i % n_batches_per_epoch == 0:
            epoch_stats = f"Finished epoch {int(i/n_batches_per_epoch)+1} | Train Loss: {train_loss:.5f}"    

            y_error = jnp.nan
            # test data evaluation
            if test_data:

                
                test_pred_ys, test_pred_dys = vmap(jax.value_and_grad(weighted_model))(test_data.x)
                y_error = jnp.sqrt(mse(test_pred_ys, test_data.y))
                dy_error = jnp.sqrt(mse(test_pred_dys, test_data.dy))
                    
                # comparing to full hessian
                test_pred_ddys = vmap(jax.hessian(MakeScalar(weighted_model)))(test_data.x)
                test_pred_ddys = test_pred_ddys.reshape(test_data.ddy.shape)
                ddy_error = jnp.sqrt(mse(test_pred_ddys, test_data.ddy))
                if variant == "3rdBatchSVD":
                    # comparing to full third derivative tensor
                    test_pred_dddys = vmap(jax.jacfwd(jax.hessian(MakeScalar(weighted_model))))(test_data.x)
                    test_pred_dddys = test_pred_dddys.reshape(test_data.dddy.shape)
                    dddy_error = jnp.sqrt(mse(test_pred_dddys, test_data.dddy))
                else: dddy_error = .0

                # 2nd order error in proj dirs
                if not variant in ["value", "1st", "fullHessian"]:

                    # comparing along directions used in loss
                    U = iteration_data["directions"]

                    b = test_data.x.shape[0]
                    d = ref_model.n_dims
                    H_true = test_data.ddy.reshape(b, d, d)
                    H_pred = test_pred_ddys.reshape(b, d, d)

                    if variant == "batchSVD" or variant == "random":
                        # batch-shared directions: U_norm  (k, d)
                        HU_true  = jnp.einsum('bij,kj->bki', H_true, U)   # (B, k, d)
                        HU_pred  = jnp.einsum('bij,kj->bki', H_pred, U)   # (B, k, d)

                    elif variant == "perXSVD" or variant == "streaming":
                        ## per-x directions: U_norm_perx  (B_train, k, d)

                        U_stack = U.reshape(-1, H_true.shape[-1])  # (B_train*k0, d)

                        # L2-normalize rows
                        eps = 1e-8
                        U_stack = U_stack / (jnp.linalg.norm(U_stack, axis=1, keepdims=True) + eps)

                        # Orthonormalize to get a shared basis U_shared: (m, d), with m <= d
                        Q, _ = jnp.linalg.qr(U_stack.T)   # (d, m)
                        U_shared = Q.T                    # (m, d)

                        # Projected HVP errors on test set using this shared basis
                        HU_true = jnp.einsum('bij,kj->bki', H_true, U_shared)   # (B_test, m, d)
                        HU_pred = jnp.einsum('bij,kj->bki', H_pred, U_shared)
                    else:
                        HU_true = jnp.nan
                        HU_pred = jnp.nan

                    proj_hvp_rmse = rmse(HU_pred, HU_true)

                else:
                    proj_hvp_rmse = jnp.nan



                epoch_stats += f" | Test Value Loss: {y_error:.5f}"
                #print("ddy error test:", ddy_error)

                
            print(epoch_stats)
        
        iteration_data["test value loss"] = y_error
        iteration_data["test grad loss"] = dy_error
        iteration_data["test hess loss"] = ddy_error
        iteration_data["test t3 loss"] = dddy_error
        iteration_data["test proj hess loss"] = proj_hvp_rmse 
        iteration_data["train loss"] = train_loss
        iteration_datas.append(iteration_data)


        #if i % (n_batches_per_epoch*10) == 0:
        #    epoch_percent = i/n_steps
    

    avg_time_per_batch = sum_batch_times / n_steps
    print(f"Average execution time per batch: {avg_time_per_batch:.5f}s")
    
    return weighted_model, iteration_datas, sketch, avg_time_per_batch