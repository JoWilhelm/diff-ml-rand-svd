import jax
import jax.random as jrandom
import jax.numpy as jnp
from jax import vmap
import equinox as eqx

from diff_ml.reference_models.analytic import Analytic
from diff_ml.reference_models.bachelier import Bachelier
from diff_ml.reference_models.heston import Heston
from diff_ml.reference_models.mnist import MNIST_ref
from diff_ml.nn.utils import init_linear_weight, trunc_init
from diff_ml.nn.train import train
from diff_ml.utils import Range
from diff_ml.losses.directions import StreamingHessianSketch
from diff_ml.utils import rmse, MakeScalar

import optax



print(jax.devices())


key = jrandom.PRNGKey(42)
key, subkey = jrandom.split(key)


#min_x, max_x, type = -5.0, 5.0, "RHE"
##min_x, max_x, type = -1.0, 1.0, "CubicRankR"
##min_x, max_x, type = -1.5, 2.5, "Rosenbrock"
##min_x, max_x, type = -5.0, 5.0, "Ackley"
##min_x, max_x, type = -5.12, 5.12, "Rastrigin"
#ref_model = Analytic(
#    key=key,
#    type=type,
#    d=7,
#    min_x=min_x,
#    max_x=max_x
#)

#ref_model = MNIST_ref(
#    key=key, 
#    scale=0.5,
#    target_class=9
#)

#basket_dim = 7
#ref_model = Heston(
#    key = key,
#    basket_dim=basket_dim,
#    basket_weights=jrandom.uniform(subkey, shape=(basket_dim,), minval=1.0, maxval=10.0),
#)

basket_dim = 7
n_paths = 0 # number of MC paths per label. Set to 0 to use analytic formula.
ref_model = Bachelier(
    subkey,
    basket_dim=basket_dim, 
    weights=jrandom.uniform(subkey, shape=(basket_dim,), minval=1.0, maxval=10.0),
    n_paths_per_label=n_paths
)



test_set = ref_model.get_test_set(n_samples=1*1024, order=3)
print("")
print("test set shapes:")
print("x:", test_set.x.shape)
print("y:", test_set.y.shape)
print("dydx:", test_set.dy.shape)
print("ddyddx:", "-" if test_set.ddy is None else test_set.ddy.shape)
print("dddydddx:", "-" if test_set.dddy is None else test_set.dddy.shape)
print("")




#variant = "value"
#variant = "1st"
#variant = "random"
variant = "batchSVD"
#variant = "3rdBatchSVD"
#variant = "perXSVD"
#variant = "streaming"
#variant = "fullHessian"


k = 1
streaming_r = 2
oversamppling_p = streaming_r - k
power_iteration_q = 0

learnable_loss_weights = True
do_approx_metrics = False


n_epochs = 10
n_batches_per_epoch = 64
batch_size = 128
lr = 1e-3



# NN surrogate model and optimizer
input_dims = ref_model.n_dims
key, subkey = jax.random.split(key)
mlp = eqx.nn.MLP(key=subkey, in_size=input_dims, out_size="scalar", width_size=20, depth=3, activation=jax.nn.silu)
key, subkey = jax.random.split(key)
mlp = init_linear_weight(mlp, trunc_init, subkey)
surrogate_model = mlp
optim = optax.adam(learning_rate=lr)



if variant == "streaming":
    # sketch
    key, subkey = jax.random.split(key)
    sketch = StreamingHessianSketch(
                ref_model=ref_model,
                r=streaming_r,
                k=k, 
                key=subkey)
else:
    sketch = None



weighted_model, iteration_datas, sketch, avg_time_per_batch = train(
                        model = surrogate_model, 
                        test_data=test_set,
                        optim=optim, 
                        n_epochs=n_epochs,
                        n_batches_per_epoch=n_batches_per_epoch,
                        batch_size=batch_size,
                        ref_model=ref_model,
                        sketch=sketch,
                        variant=variant,
                        k=k,
                        p=oversamppling_p,
                        q=power_iteration_q,
                        learnable_loss_weights=learnable_loss_weights,
                        do_approx_metrics=do_approx_metrics
                        )



# final test set evaluations
test_pred_ys, test_pred_dys = vmap(jax.value_and_grad(weighted_model))(test_set.x)
test_pred_ddys = vmap(jax.hessian(MakeScalar(weighted_model)))(test_set.x)
test_pred_dddys = vmap(jax.jacfwd(jax.hessian(MakeScalar(weighted_model))))(test_set.x)
print("")
print("test set predictions shapes:")
print("test_pred_ys shape: ", test_pred_ys.shape)
print("test_pred_dys shape: ", test_pred_dys.shape)
print("test_pred_ddys shape: ", test_pred_ddys.shape)
print("test_pred_dddys shape: ", test_pred_dddys.shape)
print("")

y_error = rmse(test_pred_ys, test_set.y)
dy_error = rmse(test_pred_dys, test_set.dy)
ddy_error = rmse(test_pred_ddys, test_set.ddy)
if test_set.dddy is not None:
    dddy_error = rmse(test_pred_dddys, test_set.dddy)
else:
    dddy_error = jnp.nan

print(f"test y error: {y_error:.5f}")
print(f"test dy error: {dy_error:.5f}")
print(f"test ddy error: {ddy_error:.5f}")
if test_set.dddy is not None:
    print(f"test dddy error: {dddy_error:.5f}")



if do_approx_metrics:
    eF_mean  = jnp.array([iteration_datas[t]["approximation metrics ref"]["eF mean"]  for t in range(n_epochs)]).mean()
    e2_mean  = jnp.array([iteration_datas[t]["approximation metrics ref"]["e2 mean"]  for t in range(n_epochs)]).mean()
    evr_mean = jnp.array([iteration_datas[t]["approximation metrics ref"]["evr mean"] for t in range(n_epochs)]).mean()
    eng_mean = jnp.array([iteration_datas[t]["approximation metrics ref"]["eng mean"] for t in range(n_epochs)]).mean()
    print("")
    print("Hessian approximation metrics for the second-order supervision directions:")
    print(f"mean error Frobenius norm: {eF_mean:.3f}")
    print(f"mean error spectral norm: {e2_mean:.3f}")
    print(f"mean explained variance ratio: {evr_mean:.3f}")
    print(f"mean energy captured: {eng_mean:.3f}")