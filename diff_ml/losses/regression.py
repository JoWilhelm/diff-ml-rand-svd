
import jax
from jax import vmap
from jax import random as jrandom
import equinox as eqx

from typing_extensions import TypeAlias
from jaxtyping import Array, Float

Data: TypeAlias = dict[str, Float[Array, "n_samples ..."]]

import jax
import jax.random as jrandom

from diff_ml.utils import mse, rmse, MakeScalar, generate_random_vectors

from diff_ml.losses.directions import get_rand_SVD_directions, get_rand_SVD_directions_per_x, get_3rd_rand_SVD_directions
from diff_ml.hvps_and_t3vps import hvp_batch, tvp_batch, hvp_batch_per_input

#print(jax.devices())




def standard_loss_fn(model, batch):
    x = batch[0]
    y = batch[1]
    y_pred = vmap(model)(x)
    return mse(y, y_pred)



def first_order_loss_fn(model, batch):
    
    x, y, dydx, additional_args = batch

    y_pred, dydx_pred = vmap(eqx.filter_value_and_grad(MakeScalar(model)))(x)
    assert(y_pred.shape == y.shape)
    assert(dydx_pred.shape == dydx.shape)

    value_loss = mse(y_pred, y)
    grad_loss = mse(dydx_pred, dydx)

    loss = 0.5*value_loss + 0.5*grad_loss
    return loss
    


@eqx.filter_jit
def second_order_loss_fn(model: eqx.nn.MLP, batch, key, ref_model, dirs, dirs_per_x, Svals, variant, k) -> Float:
    
    if not variant in ["random", "batchSVD", "perXSVD", "streaming", "fullHessian"]:
        raise ValueError("variant must be either random, batchSVD, perXSVD, streaming or fullHessian")

    iteration_data = {}

    #x = batch[0]
    #y = batch[1]
    #dydx = batch[2]
    x, y, dydx, additional_args = batch
    b = x.shape[0]
    

    y_pred, dydx_pred = vmap(eqx.filter_value_and_grad(MakeScalar(model)))(x)
    assert(y_pred.shape == y.shape)
    assert(dydx_pred.shape == dydx.shape)

    #value_loss = mse(y_pred, y)
    #grad_loss = mse(dydx_pred, dydx)    
    
    
    k = min(k, ref_model.n_dims)  # ensure k does not exceed the number of dimensions


    

    ref_fn = ref_model.reference_fn(additional_args)

    
    
            

    x_raw = x
    


    

    ## random directions
    if variant == "random":
        key, subkey = jrandom.split(key)
        rand_directions = generate_random_vectors((k, *ref_model.un_flattened_shape), key=key, normalize=True)


    # randSVD directions
    # TODO calculate this once before?

    if variant == "batchSVD": 
        rand_svd_directions_ref, eval_dir, k_dir, Svals = get_rand_SVD_directions(
                                        ref_model=ref_model,
                                        f=ref_fn,
                                        x=x_raw,
                                        k=k,
                                        key=key,
                                        is_ref_fn=True
                                        )
        
    if variant == "perXSVD":
        U_batch, eval_dir_batch, k_dir_batch, Svals = get_rand_SVD_directions_per_x(
                                        ref_model=ref_model,
                                        f=ref_fn,
                                        X=x_raw,
                                        k=k, 
                                        key=key,
                                        is_ref_fn=True
                                        )
    
    #rand_svd_directions_model, eval_dir, k_dir, S_var_model = get_rand_SVD_directions(
    #                                f=MakeScalar(model),  
    #                                x=x,
    #                                k=k,
    #                                key=key,
    #                                is_ref_fn=False
    #                                )
    
    # TODO the stacking logic seems wrong
    #stacked = jnp.stack([rand_svd_directions_ref, rand_svd_directions_model], axis=2)        
    #interleaved_rand_svd_dirs = stacked.reshape((2*len(rand_svd_directions_ref), *ref_model.un_flattened_shape))
    #combined = jnp.concatenate([rand_svd_directions_ref, rand_svd_directions_model], axis=0)

    
    if variant == "perXSVD":
        dirs_per_x = U_batch
        
    if variant == "perXSVD" or variant == "streaming":
        dirs_per_x_unflat = dirs_per_x.reshape(b, k, *ref_model.un_flattened_shape)
        dirs_per_x_flat = dirs_per_x.reshape(b, k, ref_model.n_dims)
        dirs_per_x_scaled_flat = dirs_per_x_flat
        
        

        
        ##jax.debug.print("dirs.shape {shape}", shape=directions.shape)
        #
        # targets
        target_hvps = hvp_batch_per_input(
            f=ref_fn,
            inputs=x_raw, 
            directions=dirs_per_x_scaled_flat
        )
        #jax.debug.print("targets.shape {shape}", shape=target_hvps.shape)


        # predictions
        pred_hvps = hvp_batch_per_input(
            f=MakeScalar(model),
            inputs=x, 
            directions=dirs_per_x_flat
        )
        #jax.debug.print("preds.shape {shape}", shape=pred_hvps.shape)


    
    if variant == "batchSVD" or variant == "random":
        if variant == "random":
            directions = rand_directions
        if variant == "batchSVD":
            directions = rand_svd_directions_ref
            #directions = dirs
            #directions = rand_directions
            #directions = rand_svd_directions_model
            #directions = interleaved_rand_svd_dirs
            #directions = combined

        k = directions.shape[0]

        #dirs_raw_unflat = directions                         
        #dirs_norm_unflat = dirs_raw_unflat / ref_model.x_std 
        #
        #dirs_raw_flat  = dirs_raw_unflat.reshape(k, ref_model.n_dims)
        #dirs_norm_flat = dirs_norm_unflat.reshape(k, ref_model.n_dims)

        # scaling directions to account for normalization
        dirs_flat = directions.reshape(k, ref_model.n_dims)
        dirs_scaled_flat = dirs_flat
        

        ###### ---- Second-Order Targets via CFD ---- ####
        ## prepare cfd fn
        #payoff_fn = partial(ref_model.antithetic_payoff, # TODO make loss function independent of Bachelier, pass payoff_fn
        #                    weights=ref_model.weights,
        #                    strike_price=ref_model.strike_price
        #                    )
        #D_payoff_fn = jax.vmap(jax.grad(payoff_fn)) 
        #h = 1e-1
        #paths1 = additional_args
        #cfd_of_dpayoff_fn = cfd_fn(D_payoff_fn, h, x, paths1) 
        #
        #
        #ddpayoff = jax.vmap(cfd_of_dpayoff_fn)(directions) 
        #ddpayoff = jnp.transpose(ddpayoff, (1, 0, 2)) # (batch_size, n_directions, n_dims)
        #target_hvps = ddpayoff



        #### ---- Second-Order Targets via HVPs ---- ####

        #jax.debug.print("dirs_scaled_flat.shape {shape}", shape=dirs_scaled_flat.shape)
        #jax.debug.print("x_raw.shape {shape}", shape=x_raw.shape)
        #return .0




        # all directions
        target_hvps = hvp_batch(
            f=ref_fn,
            inputs=x_raw, 
            directions=dirs_scaled_flat
        )

    

        ## conditional directions
        #target_hvps = hvp_batch_cond(f=ref_model.closed_form_basket_price_x, 
        #                                inputs=x_raw_flat, 
        #                                directions=dirs_scaled_flat,
        #                                eval_hvp=eval_dir,
        #                                )

        ## CFD
        #D_payoff_fn = jax.vmap(jax.grad(ref_model.closed_form_basket_price_x))
        ## central finite differences derivative
        #h = 1e-1
        #cfd_of_dpayoff_fn = cfd_fn(D_payoff_fn, h, x_raw_flat) 
        ## conditional directions
        #cfd_of_dpayoff_cond_fn = cfd_cond_fn(cfd_of_dpayoff_fn, batch_size=x.shape[0]) # TODO get rid of the explicit batch size dependency
        #ddpayoff_cond = cfd_of_dpayoff_cond_fn(dirs_scaled_flat, eval_dir)
        #target_hvps = jnp.transpose(ddpayoff_cond, (1, 0, 2))
        ##jax.debug.print("ddpayoff_cond[{i}] {v}", i=i, v=ddpayoff_cond[i]) 




    


        #### ---- Second-Order Predicitons via HVPs ---- ####

        # all directions
        pred_hvps = hvp_batch(
            f=MakeScalar(model),
            inputs=x, 
            directions=dirs_flat
        )

        ## conditional directions
        #pred_hvps = hvp_batch_cond(f=MakeScalar(model), 
        #                                inputs=x, 
        #                                directions=directions.reshape(k, 2*basket_dim),
        #                                eval_hvp=eval_dir,
        #                                )
        

    # (batch_size, k, basket_dim*2)
    #print("pred_hvps shape:", pred_hvps.shape)

    #print("target nans:", jnp.isnan(target_hvps).sum())
    #print("pred nans:", jnp.isnan(target_hvps).sum())


   
    if variant == "fullHessian":
        ## Ture Hessians vis jax.hessian for testing
        model_ddyddx = vmap(jax.hessian(MakeScalar(model)))(x)
        #model_ddyddx_diag = jnp.stack([model_ddyddx[:,i,: ,i,:] for i in range(basket_dim)], axis=1)
        true_ddyddx = vmap(jax.hessian(ref_fn))(x_raw)
        
        #true_ddyddx_diag = jnp.stack([true_ddyddx[:,i,: ,i,:] for i in range(basket_dim)], axis=1)
        #print("target nans:", jnp.isnan(true_ddyddx).sum())
        #print("pred nans:", jnp.isnan(model_ddyddx).sum())
        target_hvps = true_ddyddx
        pred_hvps = model_ddyddx

        #jax.debug.print("true_ddyddx shape {shape}", shape=true_ddyddx.shape)
        #jax.debug.print("pred_ddyddx shape {shape}", shape=model_ddyddx.shape)





    assert(target_hvps.shape == pred_hvps.shape)
    


    #print("ddyddx_pred shape: ", ddyddx_pred.shape)
    #print("")
    #return .0

    

    #curv = jnp.linalg.norm(target_hvps, axis=-1)              # (B,k)
    #jax.debug.print("mean dir-1st-deriv: {}", curv.mean())
    #jax.debug.print("median dir-2nd-deriv: {}", jnp.median(curv))


    hess_loss = mse(pred_hvps, target_hvps) 
    #hess_loss = cosine_loss(pred_hvps, target_hvps)
    
    ## WSE
    #if not variant == "random":
    #    hess_loss = wse(pred_hvps, target_hvps, Svals)
    #else:
    #    hess_loss = mse(pred_hvps, target_hvps)

    

    #jax.debug.print("value loss: {}", value_loss)
    #jax.debug.print("grad loss: {}", grad_loss)
    #jax.debug.print("hess loss: {}", hess_loss)
    #jax.debug.print("---------------------------------")




    if variant == "fullHessian":
        return hess_loss, iteration_data



    
    #iteration_data["directions model"] = rand_svd_directions_model
    #iteration_data["directions ref"] = rand_svd_directions_ref
    
    #iteration_data["S_var model"] = S_var_model
    #iteration_data["S_var ref"] = S_var_ref
    
    
    
    if variant == "batchSVD" or variant == "random" or variant == "3rdBatchSVD":
        iteration_data["directions"] = dirs_flat
    
    if variant == "perXSVD" or variant == "streaming":
        iteration_data["directions"] = dirs_per_x_flat

    
    # TODO move this out of 2nd order loss function
    
    #x_raw_flat = x_raw.reshape(x.shape[0], ref_model.n_dims)
    #
    #if variant == "batchSVD" or variant == "random" or variant == "3rdBatchSVD":
    #    iteration_data["directions"] = dirs_flat
    #    approx_dirs = dirs_scaled_flat
    #    if variant == "random":
    #        approx_dirs = rand_directions.reshape(-1, ref_model.n_dims)
    #    approximation_metrics_ref = approx_metrics(
    #                                               #fn=MakeScalar(model),
    #                                               fn=ref_fn,
    #                                               ref_model=ref_model,
    #                                               #x=x,
    #                                               x=x_raw_flat, 
    #                                               U_dirs=approx_dirs
    #                                               )
    #if variant == "perXSVD" or variant == "streaming":
    #    iteration_data["directions"] = dirs_per_x_flat
    #    approx_dirs_per_x = dirs_per_x_scaled_flat
    #    if variant == "random":
    #        approx_dirs_per_x = rand_directions.reshape(-1, ref_model.n_dims)
    #    approximation_metrics_ref = approx_metrics_per_x(
    #                                               #fn=MakeScalar(model),
    #                                               fn=ref_fn,
    #                                               ref_model=ref_model,
    #                                               #x=x,
    #                                               x=x_raw_flat, 
    #                                               dirs_per_x=approx_dirs_per_x
    #                                               )
    #
    #
    #iteration_data["approximation metrics ref"] = approximation_metrics_ref


    return hess_loss, iteration_data










@eqx.filter_jit
def third_order_loss_fn(model: eqx.nn.MLP, batch, key, ref_model, U_H, k) -> Float:
    
    
    x, y, dydx, additional_args = batch
    

    
    k = min(k, ref_model.n_dims)  # ensure k does not exceed the number of dimensions


    
    ref_fn = ref_model.reference_fn(additional_args)

    
    
    x_raw = x

    dirs_scaled_flat = U_H
    


    # get direction pairs
    dirs_v, dirs_w = get_3rd_rand_SVD_directions(
                                    ref_model=ref_model,
                                    f=ref_fn,
                                    x=x_raw,
                                    U_H=dirs_scaled_flat,
                                    k=k, 
                                    key=key
                                    )

    # tvps targets
    target_tvps = tvp_batch(
        f=ref_fn,
        inputs=x_raw, 
        v_dirs=dirs_v,
        w_dirs=dirs_w
    )

    # tvp predictions
    # all directions
    pred_tvps = tvp_batch(
        f=MakeScalar(model),
        inputs=x, 
        v_dirs=dirs_v,
        w_dirs=dirs_w
    )


    # tloss
    t3_loss = mse(pred_tvps, target_tvps)
        

    return t3_loss




