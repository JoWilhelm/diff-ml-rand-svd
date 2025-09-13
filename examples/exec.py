import jax
from jax import vmap
import equinox as eqx
import jax.random as jrandom
from diff_ml.reference_models.analytic import Analytic
from diff_ml.reference_models.bachelier import Bachelier
from diff_ml.reference_models.heston import Heston
from diff_ml.reference_models.mnist import MNIST_ref
from diff_ml.nn.utils import init_linear_weight, trunc_init, MakeScalar
import optax
from diff_ml.nn.train import train
from diff_ml.utils import Range
from diff_ml.losses.directions import StreamingHessianSketch

from diff_ml.utils import rmse


key = jrandom.PRNGKey(1)
key, subkey = jrandom.split(key)



#ref_model_analytic = Analytic(
#    key=key,
#    type="RHE",
#    #type="Rastrigin",
#    #type="Rosenbrock",
#    #type="Ackley", 
#    d=5,
#    min_x=-5.0,
#    max_x=5.0
#)
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
#    spot_range = Range(50.0, 150.0),
#    vol_range = Range(0.01, 0.1),
#    mc_time_steps=1024,
#    K= 100.0,
#    r = 0.00,
#    T = 1.0,
#    rho = -0.3,
#    kappa = 1.0,
#    theta = 0.09,
#    xi = 1.0
#)

basket_dim = 7
ref_model_bachelier = Bachelier(
    key,
    basket_dim=basket_dim, 
    weights=jrandom.uniform(subkey, shape=(basket_dim,), minval=1.0, maxval=10.0)
)


ref_model = ref_model_bachelier
#ref_model = ref_model_heston
#ref_model = ref_model_mnist
#ref_model = ref_model_analytic





#test_set = ref_model.get_test_set(n_samples=128)
test_set = ref_model.get_test_set(n_samples=1*1024, order=3)

#if ref_model.was_normalized:
#    y_std = test_set[4]["y_std"]
#    ref_model.set_y_std(y_std)
#    print(y_std)

print("shapes:")
print("x:", test_set.x.shape)
print("y:", test_set.y.shape)
print("dydx:", test_set.dy.shape)
print("ddyddx:", test_set.ddy.shape)
print("dddydddx:", "-" if test_set.dddy == None  else test_set.dddy.shape)





###train_sample_batch = ref_model.sample(key=key, n_samples=32)
#train_sample_batch = ref_model.sample(key=key, n_samples=256)
##
##
##
##ref_model.visualize_dataset(test_set, name="Test", is_second_order=True)
##ref_model.visualize_dataset(train_sample_batch, name="Train", is_second_order=False)
##
##
##ref_model.visualize_third(x=test_set[0], dddydddx=test_set[4], name="Test")





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
                fn=ref_model.reference_fn(), 
                ref_model=ref_model,
                d=ref_model.n_dims, 
                r=streaming_r,
                k=k, 
                key=subkey)
else:
    sketch = None

# TODO need normalization layers? at least for bachelier?
## Specify the surrogate model architecture
#key, subkey = jrandom.split(key)
#mlp = eqx.nn.MLP(key=subkey, in_size=n_dims, out_size="scalar", width_size=20, depth=3, activation=jax.nn.silu) # jax.nn.silu
#key, subkey = jrandom.split(key)
#mlp = init_model_weights(mlp, jax.nn.initializers.glorot_normal(), key=subkey)
#surrogate = Normalized(
#    Normalization(x_train_mean, x_train_std), mlp, Denormalization(y_train_mean, y_train_std)
#)

optim = optax.adam(learning_rate=1e-3)
#opt_state = optim.init(eqx.filter(weighted_model, eqx.is_array))




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
test_pred_ddys = test_pred_ddys.reshape(test_set.ddy.shape)


test_pred_dddys = vmap(jax.jacfwd(jax.hessian(MakeScalar(weighted_model))))(test_set.x)
#test_pred_dddys = test_pred_dddys.reshape(test_set[4].shape)


print("Test predictions shapes:")
print("test_pred_ys shape: ", test_pred_ys.shape)
print("test_pred_dys shape: ", test_pred_dys.shape)
print("test_pred_ddys shape: ", test_pred_ddys.shape)
print("test_pred_dddys shape: ", test_pred_dddys.shape)

print("")

y_error = rmse(test_pred_ys, test_set.y)
dy_error = rmse(test_pred_dys, test_set.dy)
ddy_error = rmse(test_pred_ddys, test_set.ddy)
if not test_set.dddy == None:
    dddy_error = rmse(test_pred_dddys, test_set.dddy)

print(f"test y error: {y_error:.5f}")
print(f"test dy error: {dy_error:.5f}")
print(f"test ddy error: {ddy_error:.5f}")
if not test_set.dddy == None:
    print(f"test dddy error: {dddy_error:.5f}")

