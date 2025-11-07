import jax
import jax.numpy as jnp
from jax import random
import equinox as eqx

from jaxtyping import PRNGKeyArray

from diff_ml.utils import generate_random_vectors, safe_normalize_vectors
from diff_ml.ad import hvp_batch, t3vp_batch
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





def get_rand_SVD_directions(ref_model, f, x, k, key, kappa=0.95):
    """
    TODO
    Randomized SVD to get k top singular directions of the Hessian of f averaged over points x.
    """

    sketch_directions = generate_random_vectors(shape=(k, ref_model.n_dims), key=key, normalize=True)
   
    # build sketch Y = H @ sketch_directions
    Y = hvp_batch(
        f=f,
        inputs=x, 
        directions=sketch_directions
    ) # (b, k, d)
    Y = jnp.mean(Y, axis=0)  # (k, d)
    Y = Y.T # (d, k)    
    
    # orthonormalize Y
    Q, _ = jnp.linalg.qr(Y) # (d, k) 

    # project via HVPs
    # each row of B is H @ q_i
    B_rows = hvp_batch(
        f=f,
        inputs=x, 
        directions=Q.T
    ) # (b, k, d)
    #jax.debug.print("B_rows.shape {shape}", shape=B_rows.shape)
    B_rows = jnp.mean(B_rows, axis=0)  # (k, d)
    # TODO set B directily B = B_rows?
    B = jnp.stack(B_rows, axis=0) # (k, d)
    #jax.debug.print("B.shape {shape}", shape=B.shape)
    
    # SVD on B
    U_tilde, S, _ = jnp.linalg.svd(B, full_matrices=False) # (k, k)
    #jax.debug.print("U_tilde.shape {shape}", shape=U_tilde.shape)  



    # lift back
    U = Q @ U_tilde  # (d, k)
    U = U.T # (k, d)
    #jax.debug.print("U.shape {shape}", shape=U.shape)

    S_var = S**2 / jnp.sum(S**2)
    eval_dir = (~(jnp.cumsum(S_var) > kappa)).at[0].set(True) # make sure that at least the first principal component is always actively used
    k_dir = jnp.sum(eval_dir) # number of principal components used to explain kappa% of variance
    
    dirs = safe_normalize_vectors(U, axis=-1) # (k, d) take rows as directions

    return dirs, eval_dir, k_dir, S_var



def get_rand_SVD_directions_per_x(ref_model, f, X, k, key, kappa=0.95):
    """
    TODO
    Randomized SVD to get k top singular directions of the Hessian of f for each point in X.
    """

    b = X.shape[0]
    keys = jax.random.split(key, b)

    def single_sample_rand_svd(x, subkey):

        sketch_directions = generate_random_vectors(shape=(k, ref_model.n_dims), key=subkey, normalize=True)  # (k, d) 
   
        # build sketch Y = H @ sketch_directions
        # for a single x hvp_batch returns (1, k, d)
        Y = hvp_batch(f, x[None, :], sketch_directions)  # (1, k, d)
        Y = Y[0]  # (k, d)
        Y = Y.T   # (d, k)

        # orthonormalize Y
        Q, _ = jnp.linalg.qr(Y)  # (d, k)

        # project
        B_rows = hvp_batch(
            f=f, 
            inputs=x[None, :],
            directions= Q.T
        )[0]  # (k, d)
        B = jnp.stack(B_rows, axis=0) # (k, d)

        # SVD on B
        U_tilde, S, _ = jnp.linalg.svd(B, full_matrices=False)  # (k,k)

        # lift back
        U = Q @ U_tilde  # (d, k)
        U = U.T # (k, d)
        
        S_var = S**2 / jnp.sum(S**2)  # (k,)
        eval_dir = (~(jnp.cumsum(S_var) > kappa)).at[0].set(True)  # (k,)
        k_dir = jnp.sum(eval_dir)

        dirs = safe_normalize_vectors(U, axis=-1) # (k, d) take rows as directions
        return dirs, eval_dir, k_dir, S_var
    

    # vmap over batch
    U_batch, eval_dir_batch, k_dir_batch, S_var_batch = jax.vmap(single_sample_rand_svd)(
        X, keys
    )  # U_batch: (b, k, d)
       # S_var_batch: (b, k)

    return U_batch, eval_dir_batch, k_dir_batch, S_var_batch






class StreamingHessianSketch(eqx.Module):
    """
    TODO
    """
    
    key: PRNGKeyArray
    Y:  jnp.ndarray
    Omega: jnp.ndarray
    k: int # number of 'local' directions at each x
    r: int # number of 'global' directions r >= k
    ref_model: ReferenceModel

    # NOTE: Y and Omega need to be acceptes as input to __init__ to make the class work with dataclasses.replace
    def __init__(self, ref_model: ReferenceModel, r: int, k: int, key: PRNGKeyArray, Y=None, Omega=None):
        self.key = key
        self.r = r
        self.k = k
        #  random map, analogous to sketch directions
        if Omega == None:
            key, sk = random.split(key)
            Omega = random.normal(sk, (ref_model.n_dims, r))    
            self.Omega = Omega
        else:
            # passed on by update step
            self.Omega = Omega
        
        # streaming accumulated sketch
        if Y == None:
            self.Y = jnp.zeros((ref_model.n_dims, r))
        else:
            self.Y = Y

        self.ref_model = ref_model
    
    
    def update_batch(self, X_batch):
        """
        Update global sketch with a batch of local samples (b, d)
        """
        # sketch update: Y_new = Y + mean( H(x_t) @ Omega )
        hv = hvp_batch(self.ref_model.reference_fn(), X_batch, self.Omega.T) # (b, r, d)
        dY = jnp.mean(hv, axis=0).T # (d, r)
        #jax.debug.print("dY: {}", dY)
        # update sketch
        Y_new = self.Y + dY
        sketch_new = replace(self, Y=Y_new)
        # small (k) number of local directions per x from updated large (r) global sketch
        local_dirs, Svals = sketch_new.local_directions_batch(X_batch) 
        return sketch_new, local_dirs, Svals

    
    def local_directions_batch(self, X_batch):
        """
        Compute local Hessian singular directions for a batch X_batch (b, d)
        """

        # orhtonormalize global sketch
        Q, _ = jnp.linalg.qr(self.Y)  # (d, r)
        #jax.debug.print("U: {}", U)

        #b, d = X_batch.shape
        Bs = hvp_batch(
            f=self.ref_model.reference_fn(), 
            inputs=X_batch, 
            directions=Q.T
            )  # (b, r, d)

        # form cores
        #Bs = jnp.einsum('di,bjd->bij', Q, Bs)  # (b, r, r)

        # small SVD per sample
        Ucores, Svals, _ = jax.vmap(lambda B_i: jnp.linalg.svd(B_i, full_matrices=False))(Bs)
        
        # truncate to top k
        Ucores_k = Ucores[:, :, :self.k]  # (b, r, k)
        
        # lift back
        Us = jnp.einsum('dr,brk->bdk', Q, Ucores_k)
        Us = Us.transpose(0, 2, 1) # (b, k, d)

        # explained variance per dir
        Svals = Svals**2
        Svals = Svals[:, :self.k]
        row_sums = jnp.sum(Svals, axis=1, keepdims=True)  # shape (b, 1)
        eps = 1e-12
        Svals = Svals / (row_sums + eps)
        #jax.debug.print("Svals shape {}", Svals.shape)
        #jax.debug.print("Svals entry 0 {}", Svals[0])
        
        local_dirs = safe_normalize_vectors(Us, axis=-1)
        return local_dirs, Svals

    





class StreamingHessianSketchOjasLite(eqx.Module):
    key: PRNGKeyArray
    Q:  jnp.ndarray      # (d, r) orthonormal basis (replace Y)
    Omega: jnp.ndarray   # (d, r) fixed probes if you still want them
    k: int
    r: int
    ref_model: ReferenceModel
    beta: float          # EMA factor, e.g. 0.1

    def __init__(self, ref_model, r, k, key, Q=None, Omega=None, C=None, beta=0.1):
        self.key = key; self.r = r; self.k = k; self.beta = beta
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



    #def update_batch(self, X_batch):
    #    eta = self.beta
    #
    #    # --- Oja residual part (exploit) ---
    #    B = hvp_batch(self.ref_model.reference_fn(), X_batch, self.Q.T)      # (b, r, d)
    #    B = jnp.transpose(B, (0, 2, 1))                                      # (b, d, r)
    #    QT_B = jnp.einsum('rd,bdj->brj', self.Q.T, B)                          # (b, r, r)
    #    proj_q = jnp.einsum('dr,brs->bds', self.Q, QT_B)                      # (b, d, r)
    #    delta_q = jnp.mean(B - proj_q, axis=0)                                # (d, r)
    #
    #    # --- small exploration on deflated Omega (explore) ---
    #    Omega_perp = self.Omega - self.Q @ (self.Q.T @ self.Omega)             # (d, r)
    #    # optional: orthonormalize probes to improve conditioning
    #    Omega_perp, _ = jnp.linalg.qr(Omega_perp)                              # (d, r)
    #    dY = hvp_batch(self.ref_model.reference_fn(), X_batch, Omega_perp.T)   # (b, r, d)
    #    delta_o = jnp.mean(jnp.transpose(dY, (0, 2, 1)), axis=0)               # (d, r)
    #
    #    # blend (epsilon small, e.g. 0.05)
    #    eps = 0.05
    #    delta = delta_q + eps * delta_o
    #
    #    # update and re-orth
    #    Q_new = self.Q + eta * delta
    #    Q_new, _ = jnp.linalg.qr(Q_new)
    #
    #    sketch_new = replace(self, Q=Q_new)
    #    local_dirs, Svals = sketch_new.local_directions_batch(X_batch)
    #    return sketch_new, local_dirs, Svals
    
    def update_batch(self, X_batch):
        eta = self.beta
    
        # --- Oja residual part (exploit) ---
        B = hvp_batch(self.ref_model.reference_fn(), X_batch, self.Q.T)      # (b, r, d)
        B = jnp.mean(B, axis=0).T  # (d, r)
        B_perp = B - self.Q @ (self.Q.T @ B)  # (d, r)
        
        ## --- small exploration on deflated Omega (explore) ---
        dY = hvp_batch(self.ref_model.reference_fn(), X_batch, self.Omega.T)   # (b, r, d)
        dy = jnp.mean(dY, axis=0).T  # (d, r)
        dy_perp = dy - self.Q @ (self.Q.T @ dy)  # (d, r)
        
        
        # blend (epsilon small, e.g. 0.05)
        eps = 0.05
        delta = B_perp + eps * dy_perp
    
        # update and re-orth
        Q_new = self.Q + eta * delta
        Q_new, _ = jnp.linalg.qr(Q_new)
    
        sketch_new = replace(self, Q=Q_new)
        local_dirs, Svals = sketch_new.local_directions_batch(X_batch)
        return sketch_new, local_dirs, Svals
    
    
    def local_directions_batch(self, X_batch):
       
        # project H onto current basis for each sample
        Bs = hvp_batch(self.ref_model.reference_fn(), X_batch, self.Q.T)  # (b, r, d)
        # TODO potentially re-use a cached B from one step before?

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

        


























def get_3rd_rand_SVD_directions(ref_model, f, x, U_H, k):
    """
    TODO
    """

    d = ref_model.n_dims
    
    ## use U_H as guided sketch directions
    sketch_directions_v = U_H[:k, :]
    sketch_directions_w = U_H[:k, :]

    # contract two modes of T with two sets of sketch directions
    Y = t3vp_batch(
        f=f,
        inputs=x, 
        v_dirs=sketch_directions_v,
        w_dirs=sketch_directions_w
    ) # (b, k, k, d)
    Y = jnp.mean(Y, axis=0)  # (k, k, d)
    Y = jnp.transpose(Y, (2, 0, 1)) # (d, k, k)

    # orthprhonormalize Y
    Y_flat = jnp.reshape(Y, (d, k*k))  # (d, k*k)
    Q, _ = jnp.linalg.qr(Y_flat, mode="reduced") # (d, q) q = min(d, k*k) 
    #jax.debug.print("Q.shape {shape}", shape=Q.shape)

    # contract the remaining mode of T
    # analogous to B = H @ Q for Hessian
    B = Q.T @ Y_flat # (q, k*k)

    # SVD on B
    # keep Vt because we are intrested in whihc pairs of sketch directions are most important
    _, S, Vt = jnp.linalg.svd(B, full_matrices=False) # (k, k)
    #jax.debug.print("U_tilde.shape {shape}", shape=U_tilde.shape)  

    # truncate to top r=k
    r = k
    S = S[:r]                         # (r,)
    V = Vt[:r, :].T                   # (k*k, r)
    V_kk_r = V.reshape(k, k, r)       # (k, k, r)

    # lift back to (d, d) matirces
    #lift right-singular directions from sketch space back to (d,d) matrices via sketch bases
    Qv_T = sketch_directions_v.T      # (d, k)
    Qw_T = sketch_directions_w.T      # (d, k)
    # first left multiply
    A = jnp.einsum('dk,kmr->dmr', Qv_T, V_kk_r) # (d, k, r)
    # then right multiply 
    Zi_all_d_r_d = jnp.einsum('imr,mj->irj', A, Qw_T.T) # (d, r, d)
    # reorder
    Zi_all = jnp.transpose(Zi_all_d_r_d, (1, 0, 2)) # (r, d, d)

    # extact leading v (left singular vector) and leading w (right singular vector) for each Zi  
    Uhat, _, VhatT = jnp.linalg.svd(Zi_all, full_matrices=False) # Uhat (r, d, d), VhatT (r, d, d)
    v_raw = Uhat[:, :, 0]          # (r, d)
    w_raw = VhatT[:, 0, :]         # (r, d)

    # Normalize each vector
    v_norm = v_raw / (jnp.linalg.norm(v_raw, axis=-1, keepdims=True) + 1e-12)
    w_norm = w_raw / (jnp.linalg.norm(w_raw, axis=-1, keepdims=True) + 1e-12)
    return v_norm, w_norm









def get_3rd_rand_SVD_directions2(ref_model, f, x, U_H, k):
    """
    TODO
    """

    d = ref_model.n_dims
    
    subspace_dirs = U_H[:k, :]  # (k, d)

    # build the restricted operator A = E[T(v_i, w_j, :)]
    A = t3vp_batch(
        f=f,
        inputs=x, 
        v_dirs=subspace_dirs,
        w_dirs=subspace_dirs
    ) # (b, k, k, d)
    A = jnp.mean(A, axis=0)  # (k, k, d)
    A = jnp.transpose(A, (2, 0, 1)) # (d, k, k)
    A_flat = jnp.reshape(A, (d, k*k))  # (d, k*k)
    
    # SVD on A_flat
    # keep Vt because we are interested in which pairs of input directions are most important
    _, _, Vt = jnp.linalg.svd(A_flat, full_matrices=False)
    #jax.debug.print("Vt.shape {shape}", shape=Vt.shape)  

    r = min(k, Vt.shape[0])
    Vt_r = Vt[:r, :]                   # (r, k*k)

    # extract leading v (left singular vector) and leading w (right singular vector) for each pair
    def _one_pair(v_pair_flat):
        M = jnp.reshape(v_pair_flat, (k, k))  # (k, k)
        Uhat, _, VhatT = jnp.linalg.svd(M, full_matrices=False) # (k, k)
        left = Uhat[:, 0]  # leading left singular vector (k,)
        right = VhatT[0, :]  # leading right singular vector (k,)

        #lift from hessian subspace to input space
        v = jnp.einsum('dk,k->d', subspace_dirs.T, left)  # (d,)
        w = jnp.einsum('dk,k->d', subspace_dirs.T, right)  # (d,)
        v = safe_normalize_vectors(v, axis=-1)
        w = safe_normalize_vectors(w, axis=-1)
        return v, w
    
    v_dirs, w_dirs = jax.vmap(_one_pair)(Vt_r)  # (r, d), (r, d)
    return v_dirs, w_dirs
