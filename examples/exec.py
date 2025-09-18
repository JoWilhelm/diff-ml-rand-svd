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


key = jrandom.PRNGKey(1)
key, subkey = jrandom.split(key)



ref_model_analytic = Analytic(
    key=key,
    type="RHE",
    #type="Rastrigin",
    #type="Rosenbrock",
    #type="Ackley", 
    d=5,
    min_x=-5.0,
    max_x=5.0
)
#RHE: min_x = -5.0, max_x = 5.0
#Rastrigin: min_x = -5.12, max_x = 5.12
#Rosenbrock: min_x = -1.5, max_x = 2.5
#Ackley: min_x = -5.0, max_x = 5.0



#ref_model_mnist = MNIST_ref(
#    key=key, 
#    scale=0.5,
#    target_class=9
#)


#basket_dim = 2
#ref_model_heston = Heston(
#    key = key,
#    basket_dim=basket_dim,
#    basket_weights=jrandom.uniform(subkey, shape=(basket_dim,), minval=1.0, maxval=10.0),
#)

#basket_dim = 7
#ref_model_bachelier = Bachelier(
#    key,
#    basket_dim=basket_dim, 
#    weights=jrandom.uniform(subkey, shape=(basket_dim,), minval=1.0, maxval=10.0)
#)


#ref_model = ref_model_bachelier
#ref_model = ref_model_heston
ref_model = ref_model_analytic
#ref_model = ref_model_mnist





#test_set = ref_model.get_test_set(n_samples=128)
test_set = ref_model.get_test_set(n_samples=1*1024, order=3)

print("shapes:")
print("x:", test_set.x.shape)
print("y:", test_set.y.shape)
print("dydx:", test_set.dy.shape)
print("ddyddx:", "-" if test_set.ddy is None else test_set.ddy.shape)
print("dddydddx:", "-" if test_set.dddy is None else test_set.dddy.shape)







#variant = "value"
#variant = "1st"
#variant = "random"
variant = "batchSVD"
#variant = "3rdBatchSVD"
#variant = "perXSVD"
#variant = "streaming"
#variant = "fullHessian"


learnable_loss_weights = True
k = 2
streaming_r = 3

n_epochs = 100  # 200 for Heston single asset convergence
n_batches_per_epoch = 32#8#32#32 # 32
BATCH_SIZE = 256#16#256#256 # 256
key = jrandom.PRNGKey(42)



# nn model
input_dims = ref_model.n_dims
key, subkey = jax.random.split(key)
mlp = eqx.nn.MLP(key=subkey, in_size=input_dims, out_size="scalar", width_size=20, depth=3, activation=jax.nn.silu)
mlp = init_linear_weight(mlp, trunc_init, subkey)
surrogate_model = mlp



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



optim = optax.adam(learning_rate=1e-3)


weighted_model, iteration_datas, sketch, avg_time_per_batch = train(
                        model = surrogate_model, 
                        test_data=test_set,
                        optim=optim, 
                        n_epochs=n_epochs,
                        n_batches_per_epoch=n_batches_per_epoch,
                        batch_size=BATCH_SIZE,
                        ref_model=ref_model,
                        sketch=sketch,
                        variant=variant,
                        k=k,
                        learnable_loss_weights=learnable_loss_weights
                        )



# eval price predictions
test_pred_ys, test_pred_dys = vmap(jax.value_and_grad(weighted_model))(test_set.x)
test_pred_ddys = vmap(jax.hessian(MakeScalar(weighted_model)))(test_set.x)
test_pred_dddys = vmap(jax.jacfwd(jax.hessian(MakeScalar(weighted_model))))(test_set.x)

print("Test predictions shapes:")
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

