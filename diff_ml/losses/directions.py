from jaxtyping import PRNGKeyArray
from typing import Callable
from diff_ml.utils import generate_random_vectors, safe_normalize_vectors
from diff_ml.hvps_and_t3vps import hvp, hvp_batch, tvp_batch

# apply PCA to first-order gradients of predictions
def PCA_of_dydx_directions(dydx, kappa=0.95, normalize=True):
    
    
    dydx_means = jnp.mean(dydx, axis=0)
    tiled_dydx_used_means = jnp.tile(dydx_means, (dydx.shape[0], 1))
    dydx_used_mean_adjusted = dydx - tiled_dydx_used_means
    U, S, VT = jnp.linalg.svd(dydx_used_mean_adjusted, full_matrices=False)
    principal_components = jnp.diag(S) @ VT
    pca_directions = principal_components.T
    #jax.debug.print("principal_components.shape {shape}", shape=principal_components.shape)
    #jax.debug.print("principal_components[0] {pc0}", pc0=principal_components[0])
    #jax.debug.print("")
    #return .0

    if normalize:
        pca_directions = normalize_vectors(pca_directions)


    # select PCs that account for kappa% of variance
    # singular values scaled to represent % of variance explained.
    S_var = S**2 / jnp.sum(S**2)
    eval_dir = (~(jnp.cumsum(S_var) > kappa)).at[0].set(True) # make use that at least the first principal component is always actively used
    k_dir = jnp.sum(eval_dir) # number of principal components used
    
    #jax.debug.print("eval_dir {v}", v=eval_dir)
    #jax.debug.print("")
    #return .0

    return pca_directions, eval_dir, k_dir





def hvp_power_iterated_sketch(f, x, sketch_directions, q):
    Y = hvp_batch(f=f, inputs=x, directions=sketch_directions) # (batch_size, k, dim)
    Y = jnp.mean(Y, axis=0)  # (k, dim)
    for _ in range(q):
        
        # --- Re-orthogonalize directions ---
        Y, _ = jnp.linalg.qr(Y.T)  # Y.T: (dim, k)
        Y = Y.T  # shape back to (k, dim)

        
        Y = hvp_batch(f=f, inputs=x, directions=Y) # (batch_size, k, dim)
        Y = jnp.mean(Y, axis=0)  # (k, dim)
        Y = hvp_batch(f=f, inputs=x, directions=Y) # (batch_size, k, dim)
        Y = jnp.mean(Y, axis=0)  # (k, dim)

    return Y



# old
def get_rand_SVD_directions(f, x, k, key, kappa=0.95, normalize=True):

    # TODO first rand svd experimental implementation
    dim = x.shape[-1]
    sketch_directions = generate_random_vectors(k, dim, key) # (k, dim)

    # Step 1: build sketch Y = H @ sketch_directions
    Y = hvp_batch(f=f, inputs=x, directions=sketch_directions) # (batch_size, k, dim)
    #jax.debug.print("Y.shape {shape}", shape=Y.shape)
    # TODO understand if averaging over batch_size is the correct approach
    Y = jnp.mean(Y, axis=0)  # (k, dim)
    Y = Y.T # (dim, k)
    #jax.debug.print("Y.shape {shape}", shape=Y.shape)

    ## power iterated version of step 1
    #Y = hvp_power_iterated_sketch(f=f, x=x, sketch_directions=sketch_directions, q=3) # (k, dim)
    #Y = Y.T # (dim, k)
    
    
    
    # Step 2: orthonormalize Y
    # TODO breaks when k > dim, which I guess makes sense
    Q, _ = jnp.linalg.qr(Y) # (dim, k)  
    #jax.debug.print("Q.shape {shape}", shape=Q.shape)

    # Step 3: each row of B is H @ q_i
    B_rows = hvp_batch(f=f, inputs=x, directions=Q.T) # (batch_size, k, dim)
    #jax.debug.print("B_rows.shape {shape}", shape=B_rows.shape)  
    # TODO understand if averaging over batch_size is the correct approach
    B_rows = jnp.mean(B_rows, axis=0) # (k, dim)
    #jax.debug.print("B_rows.shape {shape}", shape=B_rows.shape)
    B = jnp.stack(B_rows, axis=0) # (k, dim)
    #jax.debug.print("B.shape {shape}", shape=B.shape)

    # Step 4: SVD on B
    U_tilde, S, Vt = jnp.linalg.svd(B, full_matrices=False) # (k, k)
    #jax.debug.print("U_tilde.shape {shape}", shape=U_tilde.shape)



    # Step 5: Lift back U = Q @ U_tilde
    U = Q @ U_tilde  # (dim, k)
    #jax.debug.print("U.shape {shape}", shape=U.shape)
    #jax.debug.print("")

    S_var = S**2 / jnp.sum(S**2)
    eval_dir = (~(jnp.cumsum(S_var) > kappa)).at[0].set(True) # make use that at least the first principal component is always actively used
    k_dir = jnp.sum(eval_dir) # number of principal components used
    

    return U.T, eval_dir, k_dir







# TODO clean up
#new 
def get_rand_SVD_directions(ref_model, f, x, k, key, is_ref_fn, kappa=0.95):

    
    sketch_directions = generate_random_vectors(shape=(k, ref_model.n_dims), key=key, normalize=True)
    
   
    ## Step 1: build sketch Y = H @ sketch_directions
    Y = hvp_batch(
        f=f,
        inputs=x, 
        directions=sketch_directions
    ) # (batch, k, 2*basket)
   
    Y = jnp.mean(Y, axis=0)  # (k, dim)
    Y = Y.T # (2*basket, k)    
    
    # Step 2: orthonormalize Y
    Q, _ = jnp.linalg.qr(Y) # (*2basket, k) 



    ## Step 3: each row of B is H @ q_i
    B_rows = hvp_batch(
        f=f,
        inputs=x, 
        directions=Q.T
    ) # (batch, k, in_dim)
    #jax.debug.print("B_rows.shape {shape}", shape=B_rows.shape)

    B_rows = jnp.mean(B_rows, axis=0)  # (k, in_dimm)
    B = jnp.stack(B_rows, axis=0) # (k, in_dimm)
    #jax.debug.print("B.shape {shape}", shape=B.shape)
    

    # Step 4: SVD on B
    U_tilde, S, Vt = jnp.linalg.svd(B, full_matrices=False) # (k, k)
    #jax.debug.print("U_tilde.shape {shape}", shape=U_tilde.shape)  


    # Step 5: Lift back U = Q @ U_tilde
    U = Q @ U_tilde  # (2*basket, k)
    U = U.T # (k, 2*basket)
    #jax.debug.print("U.shape {shape}", shape=U.shape)

    
    S_var = S**2 / jnp.sum(S**2)
    eval_dir = (~(jnp.cumsum(S_var) > kappa)).at[0].set(True) # make use that at least the first principal component is always actively used
    k_dir = jnp.sum(eval_dir) # number of principal components used
    

    U_flat = U                 # (k, d) take rows as directions
    U_flat = safe_normalize_vectors(U_flat, axis=-1)   # L2 over the full vector
    #Qn, _  = jnp.linalg.qr(U_flat.T)           # (d, k) orthonormal columns
    #U_flat = Qn.T                               # (k, d) orthonormal directions
    dirs   = U_flat.reshape(k, *ref_model.un_flattened_shape)   # (k, n, 2)
    return dirs, eval_dir, k_dir, S_var




def get_rand_SVD_directions_per_x(ref_model, f, X, k, key, is_ref_fn, kappa=0.95):
    
    b, d = X.shape
    # split keys per sample
    keys = jax.random.split(key, b)

    def single_sample_rand_svd(x, subkey):
        # 1) sketch directions (omega) for this sample
        omega = generate_random_vectors(shape=(k, ref_model.n_dims), key=subkey, normalize=True)  # (k, d) 
   

        # 2) form Y = H(x) @ Ω^T via HVPs: directions are rows of Ω
        #    hvp_batch expects directions shape (k, d) and returns (1, k, d)
        Y = hvp_batch(f, x[None, :], omega)  # (1, k, d)
        Y = Y[0]  # (k, d)

        # 3) orthonormalize Y^T to get Q: shape (d, k)
        Q, _ = jnp.linalg.qr(Y.T)  # (d, k)

        # 4) form B = H(x) @ Q via HVPs: need hvp on each column of Q
        HQ = jnp.stack([hvp(f, x, Q[:, i]) for i in range(k)], axis=0)  # (k, d)

        # 5) small projected matrix B = Q^T H Q  ∈ ℝ^{k×k}
        B = Q.T @ HQ.T  # (k, k)

        # 6) SVD on B
        U_tilde, S, _ = jnp.linalg.svd(B, full_matrices=False)  # (k,k), (k,), (k,k)

        # 7) lift back: U_loc = Q @ U_tilde[:, :k] => (d, k)
        U_loc = Q @ U_tilde  # (d, k)

        # we want output shape (k, d) with directions as rows
        U = U_loc.T  # (k, d)

        # 8) spectrum metrics
        S_var = S**2 / jnp.sum(S**2)  # (k,)
        eval_dir = (~(jnp.cumsum(S_var) > kappa)).at[0].set(True)  # (k,)
        k_dir = jnp.sum(eval_dir)

        #dirs = U.reshape(k, *ref_model.un_flattened_shape)
        #dirs = safe_normalize_vectors(dirs, axis=-1)


        U_flat = U                 # (k, d) take rows as directions
        U_flat = safe_normalize_vectors(U_flat, axis=-1)   # L2 over the full vector
        #Qn, _  = jnp.linalg.qr(U_flat.T)           # (d, k) orthonormal columns
        #U_flat = Qn.T                               # (k, d) orthonormal directions
        dirs   = U_flat.reshape(k, *ref_model.un_flattened_shape)   # (k, n, 2)
    
        return dirs, eval_dir, k_dir, S_var

    # vmap over batch
    U_batch, eval_dir_batch, k_dir_batch, S_var_batch = jax.vmap(single_sample_rand_svd)(
        X, keys
    )  # U_batch: (b, k, d)
       # S_var_batch: (b, k)

    return U_batch, eval_dir_batch, k_dir_batch, S_var_batch





import jax
import jax.numpy as jnp
from jax import random, jacfwd, jacrev
from dataclasses import replace 



import equinox as eqx

class StreamingHessianSketch(eqx.Module):
    
    key: PRNGKeyArray
    fn:  Callable
    V:  jnp.ndarray  # shape (d, k)
    Omega: jnp.ndarray
    updates_count: int
    d: int
    k: int
    r: int
    ref_model: eqx.Module

    def __init__(self, fn, ref_model, d, r, k, key, V=None, updates_count=0, Omega=None):
        self.key = key
        self.d = d
        self.r = r
        self.k = k
        #  random map
        if Omega == None:
            key, sk = random.split(key)
            Omega_norm = random.normal(sk, (d, r))    
            Omega = Omega_norm
                
            self.Omega = Omega
        else:
            self.Omega = Omega
        
        # streaming accumulator
        if V == None:
            self.V = jnp.zeros((d, r)) 
        else:
            self.V = V

        self.fn = ref_model.reference_fn()
        self.ref_model = ref_model
        self.updates_count = 0
    
    
    def update_batch(self, X_batch):
        
        # sketch update: V_new = V + sum_t H(x_t) @ Omega
        hv = hvp_batch(self.fn, X_batch, self.Omega.T)      # shape (b, r, d)
        #dV = jnp.sum(hv.transpose(2,1,0), axis=2)          # shape (d, r)
        dV = jnp.mean(hv, axis=0).transpose(1, 0)           # shape (d, r)
        #jax.debug.print("dV: {}", dV)
        V_new = self.V + dV
        cnt_new = self.updates_count + 1
        # return with updated fields
        sketch_new = replace(self, V=V_new, updates_count=cnt_new)

        local_dirs, Svals = sketch_new.local_directions_batch(X_batch) 

        return sketch_new, local_dirs, Svals

    
    def local_directions_batch(self, X_batch):
        """
        Compute local Hessian singular directions for a batch X_batch (b, d)
        using U

        Returns an array of shape (b, k, d) where each slice [i] contains
        the top-k singular vectors of H(x_i) lifted to the original d-dim space.
        """

        U_raw = self.reconstruct_factors()
        #jax.debug.print("U: {}", U)

        #b, d = X_batch.shape
        hv = hvp_batch(self.fn, X_batch, U_raw.T)  # (b, r, d)
        # form cores
        S = jnp.einsum('di,bjd->bij', U_raw, hv)  # (b, r, r)
        # small SVD per sample
        Ucores, Svals, Vtcores = jax.vmap(lambda M: jnp.linalg.svd(M, full_matrices=False))(S)
        
        Svals = Svals**2
        Svals = Svals[:, :self.k]
        row_sums = jnp.sum(Svals, axis=1, keepdims=True)  # shape (b, 1)
        eps = 1e-12
        Svals = Svals / (row_sums + eps)
        #jax.debug.print("Svals shape {}", Svals.shape)
        #jax.debug.print("Svals entry 0 {}", Svals[0])
        
        # truncate to top k
        Ucores_k = Ucores[:, :, :self.k]  # (b, r, k)
        
        local_dirs = jnp.einsum('dr,brk->bdk', U_raw, Ucores_k)
        local_dirs = local_dirs.transpose(0, 2, 1) # (b, k, d)

        
        local_dirs = safe_normalize_vectors(local_dirs, axis=-1)
        return local_dirs, Svals


    def reconstruct_factors(self):
        """
        Orthonormalize V2,V3 to get U
        """
        U, _ = jnp.linalg.qr(self.V)  # (d, r)
        return U
    




def get_3rd_rand_SVD_directions(ref_model, f, x, U_H, k, key, kappa=0.95):

    d = ref_model.n_dims
    
    ## use U_H as guided sketch directions
    sketch_directions_v = U_H[:k, :]
    sketch_directions_w = U_H[:k, :]

    ## Step 1: build sketch
    Y = tvp_batch(
        f=f,
        inputs=x, 
        v_dirs=sketch_directions_v,
        w_dirs=sketch_directions_w
    ) # (batch, kv, kw, d)
    Y = jnp.mean(Y, axis=0)  # (k, k, d)
    Y = jnp.transpose(Y, (2, 0, 1)) # (d, k, k)


    Y_flat = jnp.reshape(Y, (d, k*k))  # (d, k*k)
    Q, _ = jnp.linalg.qr(Y_flat, mode="reduced") # (d, q) q = min(d, k*k) 
    #jax.debug.print("Q.shape {shape}", shape=Q.shape)


    # no second tvp pass here, just one pass with sketch directions


    B = Q.T @ Y_flat # (q, k*k)


    # Step 4: SVD on B
    U_tilde, S, Vt = jnp.linalg.svd(B, full_matrices=False) # (k, k)
    #jax.debug.print("U_tilde.shape {shape}", shape=U_tilde.shape)  


    r = k
    #U = U[:, :r]                                       # (d, r)
    S = S[:r]                                          # (r,)
    V = Vt[:r, :].T                                    # (k^2, r)  right sing. vecs as columns
    
    # --- Step 4: lift right-singular directions from sketch space to (d,d) matrices, batched ---
    # V -> (k,k,r)
    V_kk_r = V.reshape(k, k, r)                        # (k, k, r)
    Qv_T = sketch_directions_v.T                       # (d, k)
    Qw_T = sketch_directions_w.T                       # (d, k)


    #Compute Zi = Qv_T @ V_kk_r[:,:,i] @ Qw_T.T for all i in batch via einsum:
    # First left multiply: A = Qv_T @ V_kk_r -> (d, k, r)
    A = jnp.einsum('dk,kmr->dmr', Qv_T, V_kk_r)
    # Then right multiply by Qw_T.T (k,d): Zi_all has shape (d, r, d)
    Zi_all_d_r_d = jnp.einsum('imr,mj->irj', A, Qw_T.T)
    # Reorder to (r, d, d) – a batch of matrices
    Zi_all = jnp.transpose(Zi_all_d_r_d, (1, 0, 2))    # (r, d, d)

    
    # --- Step 5: batched SVD to get top rank-1 factors (v_i, w_i, sigma_i) for each Zi ---
    # jnp.linalg.svd supports batching: Uhat (r,d,d), Svals (r,d), VhatT (r,d,d)
    Uhat, Svals, VhatT = jnp.linalg.svd(Zi_all, full_matrices=False)
    # Leading singular triplets
    v_raw = Uhat[:, :, 0]          # (r, d)
    w_raw = VhatT[:, 0, :]         # (r, d)
    #sigma = Svals[:, 0]            # (r,)

    # Normalize each vector
    v_norm = v_raw / (jnp.linalg.norm(v_raw, axis=-1, keepdims=True) + 1e-12)
    w_norm = w_raw / (jnp.linalg.norm(w_raw, axis=-1, keepdims=True) + 1e-12)
    return v_norm, w_norm
