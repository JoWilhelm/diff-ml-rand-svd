import jax
import jax.numpy as jnp
from jax import random
import equinox as eqx

from jaxtyping import PRNGKeyArray

from diff_ml.utils import generate_random_vectors, safe_normalize_vectors
from diff_ml.ad import hvp_batch, t3vp_batch, t3vp
from diff_ml.reference_models.reference_model_class import ReferenceModel

from dataclasses import replace 



# apply PCA to first-order gradients of reference model
# credit Neil Kichler
def PCA_of_dydx_directions(dydx, kappa=0.95, normalize=True):
    
    # dydx: (b, d)

    dydx_means = jnp.mean(dydx, axis=0)
    tiled_dydx_used_means = jnp.tile(dydx_means, (dydx.shape[0], 1))
    dydx_used_mean_adjusted = dydx - tiled_dydx_used_means
    U, S, VT = jnp.linalg.svd(dydx_used_mean_adjusted, full_matrices=False)
    
    pca_directions = jnp.diag(S) @ VT
    #pca_directions = principal_components.T
    #jax.debug.print("principal_components.shape {shape}", shape=principal_components.shape)
    #jax.debug.print("principal_components[0] {pc0}", pc0=principal_components[0])
    #jax.debug.print("")
    #return .0

    if normalize:
        pca_directions = safe_normalize_vectors(pca_directions, axis=-1)

    # select PCs that account for kappa% of variance
    # singular values scaled to represent % of variance explained.
    S_var = S**2 / jnp.sum(S**2)
    eval_dir = (~(jnp.cumsum(S_var) > kappa)).at[0].set(True) # make use that at least the first principal component is always actively used
    k_dir = jnp.sum(eval_dir) # number of principal components used
    
    #jax.debug.print("eval_dir {v}", v=eval_dir)
    #jax.debug.print("")
    #return .0

    return pca_directions, eval_dir, k_dir





#def hvp_power_iterated_sketch(f, x, sketch_directions, q):
#    Y = hvp_batch(f=f, inputs=x, directions=sketch_directions) # (batch_size, k, dim)
#    Y = jnp.mean(Y, axis=0)  # (k, dim)
#    for _ in range(q):
#        
#        # --- Re-orthogonalize directions ---
#        Y, _ = jnp.linalg.qr(Y.T)  # Y.T: (dim, k)
#        Y = Y.T  # shape back to (k, dim)
#
#        
#        Y = hvp_batch(f=f, inputs=x, directions=Y) # (batch_size, k, dim)
#        Y = jnp.mean(Y, axis=0)  # (k, dim)
#        Y = hvp_batch(f=f, inputs=x, directions=Y) # (batch_size, k, dim)
#        Y = jnp.mean(Y, axis=0)  # (k, dim)
#
#    return Y





def get_rand_SVD_directions(ref_model, f, x, k, key, oversampling_p=0, power_iteration_q=0, kappa=0.95):
    """
    TODO
    Randomized SVD to get k top singular directions of the Hessian of f averaged over points x.
    """

    s = k + oversampling_p  # total number of sketch directions
    sketch_directions = generate_random_vectors(shape=(s, ref_model.n_dims), key=key, normalize=True)
   
    # build sketch Y = H @ sketch_directions
    Y = hvp_batch(
        f=f,
        inputs=x, 
        directions=sketch_directions
    ) # (b, s, d)
    Y = jnp.mean(Y, axis=0)  # (s, d)
    Y = Y.T # (d, s)    
    
    # orthonormalize Y
    Q, _ = jnp.linalg.qr(Y) # (d, s) 

    # project via HVPs
    # each row of B is H @ q_i
    B_rows = hvp_batch(
        f=f,
        inputs=x, 
        directions=Q.T
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
        Y = hvp_batch(f, x[None, :], sketch_directions)  # (1, s, d)
        Y = Y[0]  # (s, d)
        Y = Y.T   # (d, s)

        # orthonormalize Y
        Q, _ = jnp.linalg.qr(Y)  # (d, s)

        # project
        B_rows = hvp_batch(
            f=f, 
            inputs=x[None, :],
            directions= Q.T
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






#class StreamingHessianSketch(eqx.Module):
#    """
#    TODO
#    """
#    
#    key: PRNGKeyArray
#    Y:  jnp.ndarray
#    Omega: jnp.ndarray
#    k: int # number of 'local' directions at each x
#    r: int # number of 'global' directions r >= k
#    ref_model: ReferenceModel
#
#    # NOTE: Y and Omega need to be acceptes as input to __init__ to make the class work with dataclasses.replace
#    def __init__(self, ref_model: ReferenceModel, r: int, k: int, key: PRNGKeyArray, Y=None, Omega=None):
#        self.key = key
#        self.r = r
#        self.k = k
#        #  random map, analogous to sketch directions
#        if Omega == None:
#            key, sk = random.split(key)
#            Omega = random.normal(sk, (ref_model.n_dims, r))    
#            self.Omega = Omega
#        else:
#            # passed on by update step
#            self.Omega = Omega
#        
#        # streaming accumulated sketch
#        if Y == None:
#            self.Y = jnp.zeros((ref_model.n_dims, r))
#        else:
#            self.Y = Y
#
#        self.ref_model = ref_model
#    
#    
#    def update_batch(self, X_batch):
#        """
#        Update global sketch with a batch of local samples (b, d)
#        """
#        # sketch update: Y_new = Y + mean( H(x_t) @ Omega )
#        hv = hvp_batch(self.ref_model.reference_fn(), X_batch, self.Omega.T) # (b, r, d)
#        dY = jnp.mean(hv, axis=0).T # (d, r)
#        #jax.debug.print("dY: {}", dY)
#        # update sketch
#        Y_new = self.Y + dY
#        sketch_new = replace(self, Y=Y_new)
#        # small (k) number of local directions per x from updated large (r) global sketch
#        local_dirs, Svals = sketch_new.local_directions_batch(X_batch) 
#        return sketch_new, local_dirs, Svals
#
#    
#    def local_directions_batch(self, X_batch):
#        """
#        Compute local Hessian singular directions for a batch X_batch (b, d)
#        """
#
#        # orhtonormalize global sketch
#        Q, _ = jnp.linalg.qr(self.Y)  # (d, r)
#        #jax.debug.print("U: {}", U)
#
#        #b, d = X_batch.shape
#        Bs = hvp_batch(
#            f=self.ref_model.reference_fn(), 
#            inputs=X_batch, 
#            directions=Q.T
#            )  # (b, r, d)
#
#        # form cores
#        #Bs = jnp.einsum('di,bjd->bij', Q, Bs)  # (b, r, r)
#
#        # small SVD per sample
#        Ucores, Svals, _ = jax.vmap(lambda B_i: jnp.linalg.svd(B_i, full_matrices=False))(Bs)
#        
#        # truncate to top k
#        Ucores_k = Ucores[:, :, :self.k]  # (b, r, k)
#        
#        # lift back
#        Us = jnp.einsum('dr,brk->bdk', Q, Ucores_k)
#        Us = Us.transpose(0, 2, 1) # (b, k, d)
#
#        # explained variance per dir
#        Svals = Svals**2
#        Svals = Svals[:, :self.k]
#        row_sums = jnp.sum(Svals, axis=1, keepdims=True)  # shape (b, 1)
#        eps = 1e-12
#        Svals = Svals / (row_sums + eps)
#        #jax.debug.print("Svals shape {}", Svals.shape)
#        #jax.debug.print("Svals entry 0 {}", Svals[0])
#        
#        local_dirs = safe_normalize_vectors(Us, axis=-1)
#        return local_dirs, Svals
#
    





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



    def update_batch(self, X_batch):
        
        # exploration part
        Omega_perp = self.Omega - self.Q @ (self.Q.T @ self.Omega)  # (d, r)
        dQ_exploration = hvp_batch(self.ref_model.reference_fn(), X_batch, Omega_perp.T)   # (b, r, d)
        dQ_exploration = jnp.mean(dQ_exploration, axis=0).T  # (d, r)
        dQ_exploration_perp = dQ_exploration - self.Q @ (self.Q.T @ dQ_exploration)  # (d, r)

        # exploitation part
        dQ_exploitation = hvp_batch(self.ref_model.reference_fn(), X_batch, self.Q.T)      # (b, r, d)
        dQ_exploitation = jnp.mean(dQ_exploitation, axis=0).T  # (d, r)
        dQ_exploitation_perp = dQ_exploitation - self.Q @ (self.Q.T @ dQ_exploitation)  # (d, r)
        
        # blend
        eps = 0.05
        dQ = dQ_exploitation_perp + eps * dQ_exploration_perp
    
        # update and re-orth
        Q_new = self.Q + self.eta * dQ
        Q_new, _ = jnp.linalg.qr(Q_new)
    
        sketch_new = replace(self, Q=Q_new)
        local_dirs, Svals = sketch_new.local_directions_batch(X_batch)
        return sketch_new, local_dirs, Svals
    
    
    def local_directions_batch(self, X_batch):
       
        # project H onto current basis for each sample
        Bs = hvp_batch(self.ref_model.reference_fn(), X_batch, self.Q.T)  # (b, r, d)
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

        


























#def get_3rd_rand_SVD_directions(ref_model, f, x, U_H, k, key):
#    """
#    TODO
#    """
#
#    d = ref_model.n_dims
#    
#    ## use U_H as guided sketch directions
#    sketch_directions_v = U_H[:k, :]
#    sketch_directions_w = U_H[:k, :]
#
#    # contract two modes of T with two sets of sketch directions
#    Y = t3vp_batch(
#        f=f,
#        inputs=x, 
#        v_dirs=sketch_directions_v,
#        w_dirs=sketch_directions_w
#    ) # (b, k, k, d)
#    Y = jnp.mean(Y, axis=0)  # (k, k, d)
#    Y = jnp.transpose(Y, (2, 0, 1)) # (d, k, k)
#
#    # orthprhonormalize Y
#    Y_flat = jnp.reshape(Y, (d, k*k))  # (d, k*k)
#    Q, _ = jnp.linalg.qr(Y_flat, mode="reduced") # (d, q) q = min(d, k*k) 
#    #jax.debug.print("Q.shape {shape}", shape=Q.shape)
#
#
#
#    # contract the remaining mode of T
#    # analogous to B = H @ Q for Hessian
#    B = Q.T @ Y_flat # (q, k*k)
#
#    # SVD on B
#    # keep Vt because we are intrested in whihc pairs of sketch directions are most important
#    _, S, Vt = jnp.linalg.svd(B, full_matrices=False) # (k, k)
#    #jax.debug.print("U_tilde.shape {shape}", shape=U_tilde.shape)  
#
#    # truncate to top r=k
#    r = k
#    S = S[:r]                         # (r,)
#    V = Vt[:r, :].T                   # (k*k, r)
#    V_kk_r = V.reshape(k, k, r)       # (k, k, r)
#
#    # lift back to (d, d) matirces
#    #lift right-singular directions from sketch space back to (d,d) matrices via sketch bases
#    Qv_T = sketch_directions_v.T      # (d, k)
#    Qw_T = sketch_directions_w.T      # (d, k)
#    # first left multiply
#    A = jnp.einsum('dk,kmr->dmr', Qv_T, V_kk_r) # (d, k, r)
#    # then right multiply 
#    Zi_all_d_r_d = jnp.einsum('imr,mj->irj', A, Qw_T.T) # (d, r, d)
#    # reorder
#    Zi_all = jnp.transpose(Zi_all_d_r_d, (1, 0, 2)) # (r, d, d)
#
#    # extact leading v (left singular vector) and leading w (right singular vector) for each Zi  
#    Uhat, _, VhatT = jnp.linalg.svd(Zi_all, full_matrices=False) # Uhat (r, d, d), VhatT (r, d, d)
#    v_raw = Uhat[:, :, 0]          # (r, d)
#    w_raw = VhatT[:, 0, :]         # (r, d)
#
#    # Normalize each vector
#    v_norm = v_raw / (jnp.linalg.norm(v_raw, axis=-1, keepdims=True) + 1e-12)
#    w_norm = w_raw / (jnp.linalg.norm(w_raw, axis=-1, keepdims=True) + 1e-12)
#    return v_norm, w_norm









def get_3rd_rand_SVD_directions(ref_model, f, x, U_H, k, key):
    """
    TODO
    """

    d = ref_model.n_dims
    
    # use U_H as guided sketch directions
    seed_dirs = U_H[:k, :]  # (k, d)
    #sketch_directions_v = seed_dirs
    #sketch_directions_w = seed_dirs
    
    # TODO maybe concat with some random ones?
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
        w_dirs=sketch_directions_w
    ) # (b, r, r, d)

    #average over inputs to get E_x T(., v_i, w_j)
    Y = jnp.mean(Y, axis=0)  # (r, r, d)
    Y = jnp.transpose(Y, (2, 0, 1)) # (d, r, r)
    Y_flat = jnp.reshape(Y, (d, -1))  # (d, r*r)
    
    # orthprhonormal basis Q for range(Y)
    Q, _ = jnp.linalg.qr(Y_flat, mode="reduced") # (d, s) s = min(d, r*r) 
    #jax.debug.print("Q.shape {shape}", shape=Q.shape)
    s = Q.shape[1]


    # instead of SVD on B = Q.T T3, define energy function to rank directions in Q
    def one_energy(q):
        tvps = jax.vmap(lambda primal: t3vp(f, primal, q, q))(x)
        mean = jnp.mean(tvps, axis=0)  # (d,)
        return jnp.sum(mean**2)
    
    energies = jax.vmap(one_energy)(Q.T)  # (s,)
    ranked_indices = jnp.argsort(-energies)[:k]  # descending order
    U = Q[:, ranked_indices]  # (d, k)
    U = U.T  # (k, d)
    U = safe_normalize_vectors(U, axis=-1)
    return U, U  # return same for v and w directions


