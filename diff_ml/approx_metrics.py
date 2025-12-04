import jax
import jax.numpy as jnp


def approx_metrics(fn, x, U_dirs, batch_key):
    """
    This function computes approximation metrics for how well the given U captures the Hessian of fn at points x.
    """

    b = x.shape[0]
    keys = jax.random.split(batch_key, b)

    # Hessian at every x (b, d, d)
    H = jax.vmap(jax.hessian(fn))(x, keys)
    # NOTE: for MC approximated refernce functions this is not reliable. The ground truth H needs to be exact.

    eps = 1e-12
    fro = jnp.maximum(jnp.linalg.norm(H, axis=(1,2)), eps)
    spec = jnp.maximum(jnp.linalg.norm(H, ord=2, axis=(1,2)), eps)

    # Orthonormalize U (columns): Q: (d, k_ortho)
    Q, _ = jnp.linalg.qr(U_dirs.T)         # U_dirs: (k,d) -> Q: (d,k)
    # Projected pieces
    HU = jnp.einsum('bij,jk->bik', H, Q)   # (B, k, d)

    
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





def approx_metrics_per_x(fn, x, dirs_per_x, batch_key):
    """
    This function computes approximation metrics for how well the given per-input directions U_i capture the Hessian of fn at points x_i.
    Used for perX variant
    """
    
    b = x.shape[0]
    keys = jax.random.split(batch_key, b)

    # Hessian at every x (b, d, d)
    H = jax.vmap(jax.hessian(fn))(x, keys)
    # NOTE: for MC approximated refernce functions this is not reliable. The ground truth H needs to be exact.
    
    
    def per_sample(H_i, U_i):
        eps = 1e-12
        fro = jnp.maximum(jnp.linalg.norm(H_i), eps)
        spec = jnp.maximum(jnp.linalg.norm(H_i, ord=2), eps)
        
        # Orthonormalize (columns): Q: (d, k_i)
        Q, _ = jnp.linalg.qr(U_i.T)

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
