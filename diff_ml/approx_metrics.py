import jax
import jax.numpy as jnp


def approx_metrics(fn, ref_model, x, U_dirs):
    """
    Batch-shared U_dirs: shape (k, d) in the SAME coordinate system as x and fn.
    Returns relative Frobenius residual eF, relative spectral residual e2,
    and explained-variance ratio evr. All in [0,1] (up to tiny num. error).
    """
    b = x.shape[0]
    # Hessians (B, d, d) in this coordinate system
    H = jax.vmap(jax.hessian(fn))(x)


    #if ref_model.was_normalized:
    #    # everything from raw back to normalized space
    #    H = H.reshape(b, *ref_model.un_flattened_shape, *ref_model.un_flattened_shape)
    #
    #    scale = jnp.tensordot(ref_model.x_std, ref_model.x_std, axes=0) / ref_model.y_std
    #    H = H * scale
    #    H = H.reshape(b, ref_model.n_dims, ref_model.n_dims)
    #
    #    U_dirs = U_dirs.reshape(-1, *ref_model.un_flattened_shape)
    #    U_dirs = U_dirs / ref_model.x_std
    #    U_dirs = U_dirs.reshape(-1, ref_model.n_dims)


    
    #B, d, _ = H.shape
    #k = U_dirs.shape[0]

    # Orthonormalize U (columns): Q: (d, k_ortho)
    Q, _ = jnp.linalg.qr(U_dirs.T)         # U_dirs: (k,d) -> Q: (d,k)
    # Projected pieces
    HU = jnp.einsum('bij,jk->bik', H, Q)   # (B, k, d)

    fro = jnp.linalg.norm(H, axis=(1,2))
    spec = jnp.linalg.norm(H, ord=2, axis=(1,2))
    eps = 1e-12
    fro = jnp.maximum(fro, eps)
    spec = jnp.maximum(spec, eps)

    HU_fro_sq = jnp.sum(HU**2, axis=(1,2))               # ||H Q||_F^2
    evr = HU_fro_sq / (fro**2)                           # ∈ [0,1]

    # Right-projection residual: H - H P, with P = Q Q^T
    P = Q @ Q.T                                          # (d,d)
    residual = H - jnp.einsum('bij,jl->bil', H, P)       # H(I-P)
    eF = jnp.linalg.norm(residual, axis=(1,2)) / fro     # ∈ [0,1]
    e2 = jnp.linalg.norm(residual, ord=2, axis=(1,2)) / spec

    total = jnp.sum(H**2, axis=(1,2))
    captured = jnp.sum((H @ Q)**2, axis=(1,2))
    energy_fraction = jnp.mean(captured / (total + 1e-12))

    return {
        "eF mean": jnp.mean(eF),
        "e2 mean": jnp.mean(e2),
        "evr mean": jnp.mean(evr),
        "eng mean": energy_fraction,
    }

def approx_metrics_per_x(fn, ref_model, x, dirs_per_x):
    """
    Per-x U: dirs_per_x (B, k, d) in SAME coords as x and fn.
    """
    H = jax.vmap(jax.hessian(fn))(x)                     # (B,d,d)
    #B, k, d = dirs_per_x.shape[0], dirs_per_x.shape[1], dirs_per_x.shape[2]
    eps = 1e-12


    #b = x.shape[0]
    #if ref_model.was_normalized:
    #    # everything from raw back to normalized space
    #    H = H.reshape(b, *ref_model.un_flattened_shape, *ref_model.un_flattened_shape)
    #
    #    scale = jnp.tensordot(ref_model.x_std, ref_model.x_std, axes=0) / ref_model.y_std
    #    H = H * scale
    #    H = H.reshape(b, ref_model.n_dims, ref_model.n_dims)
    #
    #    dirs_per_x = dirs_per_x.reshape(-1, *ref_model.un_flattened_shape)
    #    dirs_per_x = dirs_per_x / ref_model.x_std
    #    dirs_per_x = dirs_per_x.reshape(-1, ref_model.n_dims)


    def per_sample(H_i, U_i):
        # Orthonormalize (columns): Q: (d, k_i)
        Q, _ = jnp.linalg.qr(U_i.T)
        fro = jnp.maximum(jnp.linalg.norm(H_i), eps)
        spec = jnp.maximum(jnp.linalg.norm(H_i, ord=2), eps)

        HU = H_i @ Q                                     # (d,k)
        HU_fro_sq = jnp.sum(HU**2)

        P = Q @ Q.T
        residual = H_i - H_i @ P                         # right projection
        eF = jnp.linalg.norm(residual) / fro
        e2 = jnp.linalg.norm(residual, ord=2) / spec
        evr = HU_fro_sq / (fro**2)
        
        total = jnp.sum(H**2, axis=(1,2))
        captured = jnp.sum((H @ Q)**2, axis=(1,2))
        energy_fraction = jnp.mean(captured / (total + 1e-12))

        return eF, e2, evr, energy_fraction


    eF, e2, evr, energy_fraction = jax.vmap(per_sample)(H, dirs_per_x)

    
    return {
        "eF mean": jnp.mean(eF), 
        "e2 mean": jnp.mean(e2), 
        "evr mean": jnp.mean(evr),
        "eng mean": jnp.mean(energy_fraction),
        }
