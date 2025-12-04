import jax
import jax.numpy as jnp
from jax import random
import equinox as eqx

from jaxtyping import PRNGKeyArray

from diff_ml.utils import generate_random_vectors, safe_normalize_vectors
from diff_ml.ad import hvp_batch, t3vp_batch, t3vp
from diff_ml.reference_models.reference_model_class import ReferenceModel

from dataclasses import replace 


# credit Neil Kichler
# apply PCA to first-order gradients of the reference model
@eqx.filter_jit
def PCA_of_dydx_directions(dydx, kappa=0.95, normalize=True):
    # dydx: (b, d)

    dydx_means = jnp.mean(dydx, axis=0)
    tiled_dydx_used_means = jnp.tile(dydx_means, (dydx.shape[0], 1))
    dydx_used_mean_adjusted = dydx - tiled_dydx_used_means
    U, S, VT = jnp.linalg.svd(dydx_used_mean_adjusted, full_matrices=False)
    
    pca_directions = jnp.diag(S) @ VT

    if normalize:
        pca_directions = safe_normalize_vectors(pca_directions, axis=-1)

    # select PCs that account for kappa% of variance
    # singular values scaled to represent % of variance explained.
    S_var = S**2 / jnp.sum(S**2)
    eval_dir = (~(jnp.cumsum(S_var) > kappa)).at[0].set(True) # make use that at least the first principal component is always actively used
    k_dir = jnp.sum(eval_dir) # number of principal components used

    return pca_directions, eval_dir, k_dir




def hvp_power_iterated_sketch(f, x, sketch_directions, q, key):
    Y = hvp_batch(f=f, 
                  inputs=x, 
                  directions=sketch_directions, 
                  batch_key=key
        )# (batch_size, k, dim)
    Y = jnp.mean(Y, axis=0)  # (k, dim)

    for _ in range(q):    
        # --- Re-orthogonalize directions ---
        Y, _ = jnp.linalg.qr(Y.T)  # Y.T: (dim, k)
        Y = Y.T  # shape back to (k, dim)

        key, subkey = jax.random.split(key)

        Y = hvp_batch(f=f, inputs=x, directions=Y, batch_key=subkey) # (batch_size, k, dim)
        Y = jnp.mean(Y, axis=0)  # (k, dim)

        # in the non-symmetric case the matrix is applied twice per step: A^T A
        #Y = hvp_batch(f=f, inputs=x, directions=Y) # (batch_size, k, dim)
        #Y = jnp.mean(Y, axis=0)  # (k, dim)
    return Y, key



@eqx.filter_jit
def get_rand_SVD_directions(ref_model, f, x, k, key, oversampling_p=0, power_iteration_q=0, kappa=0.95):
    """
    TODO
    Randomized SVD to get k top singular directions of the Hessian of f averaged over points x.
    """

    s = k + oversampling_p  # total number of sketch directions
    key, subkey = jax.random.split(key)
    sketch_directions = generate_random_vectors(shape=(s, ref_model.n_dims), key=subkey, normalize=True)
   
    key, subkey = jax.random.split(key)
    
    ## without power-iteration
    ## build sketch Y = H @ sketch_directions
    #Y = hvp_batch(
    #    f=f,
    #    inputs=x, 
    #    directions=sketch_directions,
    #    batch_key=subkey
    #) # (b, s, d)
    #Y = jnp.mean(Y, axis=0)  # (s, d)
    
    # sketch with power-iteration
    Y, key = hvp_power_iterated_sketch(
        f=f,
        x=x,
        sketch_directions=sketch_directions,
        q=power_iteration_q,
        key=subkey
    )  # (s, d)

    Y = Y.T # (d, s)    
    
    # orthonormalize Y
    Q, _ = jnp.linalg.qr(Y) # (d, s) 

    # project via HVPs
    # each row of B is H @ q_i
    key, subkey = jax.random.split(key)
    B_rows = hvp_batch(
        f=f,
        inputs=x, 
        directions=Q.T,
        batch_key=subkey
    ) # (b, s, d)
    #jax.debug.print("B_rows.shape {shape}", shape=B_rows.shape)
    B_rows = jnp.mean(B_rows, axis=0)  # (s, d)
    # TODO set B directily B = B_rows?
    B = jnp.stack(B_rows, axis=0) # (s, d)
    #jax.debug.print("B.shape {shape}", shape=B.shape)
    
    # SVD on B
    U_tilde, S, _ = jnp.linalg.svd(B, full_matrices=False) # (s, s)
    #jax.debug.print("U_tilde.shape {shape}", shape=U_tilde.shape)  

    # lift back
    U = Q @ U_tilde  # (d, s)
    U = U.T # (s, d)
    #jax.debug.print("U.shape {shape}", shape=U.shape)

    S_var = S**2 / jnp.sum(S**2)
    eval_dir = (~(jnp.cumsum(S_var) > kappa)).at[0].set(True) # make sure that at least the first principal component is always actively used
    k_dir = jnp.sum(eval_dir) # number of principal components used to explain kappa% of variance
    
    dirs = safe_normalize_vectors(U, axis=-1) # (s, d) take rows as directions
    
    dirs = dirs[:k, :]  # truncate to top k (k, d)

    return dirs, eval_dir, k_dir, S_var






@eqx.filter_jit
def get_rand_SVD_directions_per_x(ref_model, f, X, k, key, oversampling_p=0, power_iteration_q=0, kappa=0.95):
    """
    TODO
    Randomized SVD to get k top singular directions of the Hessian of f for each point in X.
    """

    b = X.shape[0]
    keys = jax.random.split(key, b)
    s = k + oversampling_p  # total number of sketch directions

    def single_sample_rand_svd(x, subkey):

        sketch_directions = generate_random_vectors(shape=(s, ref_model.n_dims), key=subkey, normalize=True)  # (s, d) 
   
        # build sketch Y = H @ sketch_directions
        # for a single x hvp_batch returns (1, s, d)
        key, subkey = jax.random.split(subkey)
        Y = hvp_batch(f, x[None, :], sketch_directions, batch_key=subkey)  # (1, s, d)
        Y = Y[0]  # (s, d)
        Y = Y.T   # (d, s)

        # orthonormalize Y
        Q, _ = jnp.linalg.qr(Y)  # (d, s)

        # project
        key, subkey = jax.random.split(key)
        B_rows = hvp_batch(
            f=f, 
            inputs=x[None, :],
            directions= Q.T,
            batch_key=subkey
        )[0]  # (s, d)
        B = jnp.stack(B_rows, axis=0) # (s, d)

        # SVD on B
        U_tilde, S, _ = jnp.linalg.svd(B, full_matrices=False)  # (s,s)

        # lift back
        U = Q @ U_tilde  # (d, s)
        U = U.T # (s, d)
        
        S_var = S**2 / jnp.sum(S**2)  # (s,)
        eval_dir = (~(jnp.cumsum(S_var) > kappa)).at[0].set(True)  # (s,)
        k_dir = jnp.sum(eval_dir)

        dirs = safe_normalize_vectors(U, axis=-1) # (s, d) take rows as directions
        
        dirs = dirs[:k, :]  # truncate to top k (k, d)

        return dirs, eval_dir, k_dir, S_var
    

    # vmap over batch
    U_batch, eval_dir_batch, k_dir_batch, S_var_batch = jax.vmap(single_sample_rand_svd)(
        X, keys
    )  # U_batch: (b, k, d)
       # S_var_batch: (b, k)

    return U_batch, eval_dir_batch, k_dir_batch, S_var_batch


    





class StreamingHessianSketch(eqx.Module):
    key: PRNGKeyArray
    Q:  jnp.ndarray      # (d, r) orthonormal basis (replace Y)
    Omega: jnp.ndarray   # (d, r) fixed probes if you still want them
    k: int
    r: int
    ref_model: ReferenceModel
    eta: float          # update step size

    def __init__(self, ref_model, r, k, key, Q=None, Omega=None, C=None, eta=0.05):
        self.key = key; self.r = r; self.k = k; self.eta = eta
        d = ref_model.n_dims
        if Q is None:
            key, sk = random.split(key)
            Q0 = random.normal(sk, (d, r))
            self.Q, _ = jnp.linalg.qr(Q0)           # (d, r)
        else:
            self.Q = Q
        if Omega is None:
            key, sk = random.split(key)
            self.Omega = random.normal(sk, (d, r))  # optional
        else:
            self.Omega = Omega
        self.ref_model = ref_model



    @eqx.filter_jit
    def update_batch(self, X_batch, batch_key):
        
        # exploration part
        Omega_perp = self.Omega - self.Q @ (self.Q.T @ self.Omega)  # (d, r)
        dQ_exploration = hvp_batch(self.ref_model.reference_fn(), X_batch, Omega_perp.T, batch_key=batch_key)   # (b, r, d)
        dQ_exploration = jnp.mean(dQ_exploration, axis=0).T  # (d, r)
        dQ_exploration_perp = dQ_exploration - self.Q @ (self.Q.T @ dQ_exploration)  # (d, r)

        # exploitation part
        dQ_exploitation = hvp_batch(self.ref_model.reference_fn(), X_batch, self.Q.T, batch_key=batch_key)      # (b, r, d)
        dQ_exploitation = jnp.mean(dQ_exploitation, axis=0).T  # (d, r)
        dQ_exploitation_perp = dQ_exploitation - self.Q @ (self.Q.T @ dQ_exploitation)  # (d, r)
        
        # blend
        eps = 0.05
        dQ = dQ_exploitation_perp + eps * dQ_exploration_perp
    
        # update and re-orth
        Q_new = self.Q + self.eta * dQ
        Q_new, _ = jnp.linalg.qr(Q_new)
    
        sketch_new = replace(self, Q=Q_new)
        local_dirs, Svals = sketch_new.local_directions_batch(X_batch, batch_key)
        return sketch_new, local_dirs, Svals
    
    
    @eqx.filter_jit
    def local_directions_batch(self, X_batch, batch_key):
       
        # project H onto current basis for each sample
        Bs = hvp_batch(self.ref_model.reference_fn(), X_batch, self.Q.T, batch_key=batch_key)  # (b, r, d)
        # TODO potentially re-use a cached B from update_batch() with one step lag instead of recompute?

        B = jnp.mean(Bs, axis=0)  # (r, d)
        #jax.debug.print("B.shape {shape}", shape=B.shape)

        # SVD on B
        U_tilde, S, _ = jnp.linalg.svd(B, full_matrices=False) # (r, r)
        #jax.debug.print("U_tilde.shape {shape}", shape=U_tilde.shape)  

        # lift back
        U = self.Q @ U_tilde  # (d, r)
        U = U.T # (r, d)
        #jax.debug.print("U.shape {shape}", shape=U.shape)

        # truncate to top k (k, r)
        U = U[:self.k, :]  # (k, d)
        S_vals = S[:self.k]  # (k,)
        S_vals = S_vals**2 / jnp.sum(S**2)

        dirs = safe_normalize_vectors(U, axis=-1) # (k, d) take rows as directions

        return dirs, S_vals

        










@eqx.filter_jit
def get_3rd_rand_SVD_directions(ref_model, f, x, U_H, k, key):
    """
    TODO
    """

    d = ref_model.n_dims
    b = x.shape[0]
    
    # use U_H as guided sketch directions
    seed_dirs = U_H[:k, :]  # (k, d)
    
    n_rand = k
    key, subkey = jax.random.split(key)
    rand_v = jax.random.normal(subkey, (n_rand, d)) if n_rand > 0 else jnp.empty((0, d))
    key, subkey = jax.random.split(key)
    rand_w = jax.random.normal(subkey, (n_rand, d)) if n_rand > 0 else jnp.empty((0, d))
    sketch_directions_v = jnp.concatenate([seed_dirs, rand_v], axis=0)
    sketch_directions_w = jnp.concatenate([seed_dirs, rand_w], axis=0)

    # contract two modes of T with two sets of sketch directions
    Y = t3vp_batch(
        f=f,
        inputs=x, 
        v_dirs=sketch_directions_v,
        w_dirs=sketch_directions_w,
        batch_key=key
    ) # (b, r, r, d)

    #average over inputs to get E_x T(., v_i, w_j)
    Y = jnp.mean(Y, axis=0)  # (r, r, d)
    Y = jnp.transpose(Y, (2, 0, 1)) # (d, r, r)
    Y_flat = jnp.reshape(Y, (d, -1))  # (d, r*r)
    
    # orthprhonormal basis Q for range(Y)
    Q, _ = jnp.linalg.qr(Y_flat, mode="reduced") # (d, s) s = min(d, r*r) 
    #jax.debug.print("Q.shape {shape}", shape=Q.shape)
    s = Q.shape[1]

    keys = jax.random.split(key, b)
    # define energy function to rank directions in Q
    def one_energy(q):
        tvps = jax.vmap(lambda primal, key: t3vp(f, primal, q, q, key), in_axes=(0, 0))(x, keys)
        mean = jnp.mean(tvps, axis=0)  # (d,)
        return jnp.sum(mean**2)
    
    energies = jax.vmap(one_energy)(Q.T)  # (s,)
    ranked_indices = jnp.argsort(-energies)[:k]  # descending order
    U = Q[:, ranked_indices]  # (d, k)
    U = U.T  # (k, d)
    U = safe_normalize_vectors(U, axis=-1)
    return U, U  # return same for v and w directions


