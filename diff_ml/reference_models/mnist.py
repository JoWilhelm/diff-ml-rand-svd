import argparse
import jax
import jax.numpy as jnp
import jax.random as random
import jax.image as jim
import equinox as eqx
import optax
import tensorflow_datasets as tfds
from jaxtyping import Array, Float, PRNGKeyArray, ScalarLike
from typing import Final
from dataclasses import field
from typing import Tuple
from jaxtyping import Array, Float, PRNGKeyArray


import jax
import jax.numpy as jnp
import equinox as eqx
import jax.random as jrandom

from typing_extensions import TypeAlias



import jax.numpy as jnp

Data: TypeAlias = dict[str, Float[Array, "n_samples ..."]]
from diff_ml.utils import Range
from diff_ml.typing import DifferentialData
from diff_ml.reference_models.reference_model_class import ReferenceModel
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

class CNN(eqx.Module):
    
    convs: list
    #pools: list
    linear: eqx.nn.Linear
    

    def __init__(self, input_size: int, depth: int, base_channels: int, num_classes: int, key):
        keys = random.split(key, depth + 1)
        self.convs = []
        #self.pools = []
        in_ch = 1
        
        for i in range(depth):
            out_ch = base_channels * (2**i)
            self.convs.append(eqx.nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, key=keys[i]))
            #self.pools.append(eqx.nn.AvgPool2d(kernel_size=2, stride=2))
            in_ch = out_ch


        # compute final spatial size
        size = input_size
        for _ in range(depth):
            size = (size - 1) // 2 + 1
            #size = size // 2 
        final_size = size
    
    
        self.linear = eqx.nn.Linear(in_ch * final_size * final_size, num_classes, key=keys[-1])
        
    def __call__(self, x):
        # x: (H, W, 1)
        x = jnp.transpose(x, (2, 0, 1))          # (C, H, W)
        for conv in self.convs:
            x = conv(x)
            x = jax.nn.silu(x)
            #x = pool(x)
        x = x.reshape(-1)                       
        return self.linear(x)



class MNIST_ref(ReferenceModel):

    key_test: PRNGKeyArray
    key_train: PRNGKeyArray
    
    train_imgs: Float[Array, "train_size H W 1"] 
    test_imgs: Float[Array, "test_size H W 1"]
    train_lbls: Float[Array, "train_size"]
    test_lbls: Float[Array, "test_size"]

    cnn: CNN
    n_dims: int
    un_flattened_shape: tuple
    scale: float = 1.0  
    target_class: int = 9  

    was_normalized: bool = False
    
    def __init__(self, key, scale, target_class=9):
        self.key_test, self.key_train, key_cnn = jax.random.split(key, 3)
        self.scale = scale
        self.target_class = target_class
        input_size = int(28 * scale)
        self.n_dims = input_size * input_size
        self.un_flattened_shape = (input_size, input_size, 1)

        self.cnn = CNN(
            input_size=int(28 * scale),
            depth=3,  # n convolutional layers
            base_channels=32, # m channels in first conv layer
            num_classes=10,
            key=key_cnn
        )

        # train the CNN
        self.cnn = self.train_CNN(
            lr=1e-3, 
            batch_size=128, 
            epochs=3
        )



    def load_mnist(self):
        print("Loading MNIST dataset...")
        # load entire train & test splits as NumPy arrays
        ds = tfds.load("mnist", split=["train", "test"],
                       batch_size=-1, as_supervised=True)
        train_imgs, train_lbls = tfds.as_numpy(ds[0])
        test_imgs,  test_lbls  = tfds.as_numpy(ds[1])

        # normalize and resize
        train_imgs = train_imgs.astype(jnp.float32) / 255.0
        test_imgs  = test_imgs .astype(jnp.float32) / 255.0

        new_size = int(28 * self.scale)
        train_imgs = jim.resize(train_imgs, (train_imgs.shape[0], new_size, new_size, 1), method="bilinear")
        test_imgs  = jim.resize(test_imgs,  (test_imgs .shape[0], new_size, new_size, 1), method="bilinear")

        train_lbls = train_lbls.astype(jnp.int32)
        test_lbls  = test_lbls .astype(jnp.int32)

        return train_imgs, train_lbls, test_imgs, test_lbls


    # smoothed labels
    def loss_fn(self, model, xb, yb, eps=0.1):
        logits = jax.vmap(model)(xb)
        num_classes = logits.shape[-1]
        soft = jax.nn.one_hot(yb, num_classes) * (1 - eps) + eps / num_classes
        return optax.softmax_cross_entropy(logits, soft).mean()



    def accuracy(self, model, xb, yb):
        preds = jnp.argmax(jax.vmap(model)(xb), axis=-1)
        return jnp.mean(preds == yb)


    def train_CNN(self, lr, batch_size, epochs):

        train_imgs, train_lbls, test_imgs, test_lbls = self.load_mnist()
        self.train_imgs = train_imgs
        self.test_imgs = test_imgs
        self.train_lbls = train_lbls
        self.test_lbls = test_lbls
        
        num_train = train_imgs.shape[0]
        num_test  = test_imgs.shape[0]

        model = self.cnn
        optimizer = optax.adam(lr)
        opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

        @eqx.filter_jit
        def train_step(model, opt_state, xb, yb):
            loss, grads = eqx.filter_value_and_grad(self.loss_fn)(model, xb, yb)
            updates, opt_state = optimizer.update(grads, opt_state, model)
            model = eqx.apply_updates(model, updates)
            return model, opt_state

        print("Training reference model...")
        for epoch in range(1, epochs+1):
            print(f"Epoch {epoch}/{epochs}")
            # a) shuffle train set
            self.key_test, perm_key = random.split(self.key_test)
            perm = random.permutation(perm_key, num_train)
            train_imgs = train_imgs[perm]
            train_lbls = train_lbls[perm]

            # b) training by slicing
            for i in range(0, num_train, batch_size):
                xb = train_imgs[i:i+batch_size]
                yb = train_lbls[i:i+batch_size]
                model, opt_state = train_step(model, opt_state, xb, yb)

            # c) evaluation
            accs = []
            for i in range(0, num_test, batch_size):
                xb = test_imgs[i:i+batch_size]
                yb = test_lbls[i:i+batch_size]
                accs.append(self.accuracy(model, xb, yb))
            test_acc = jnp.stack(accs).mean()

            print(f" → test accuracy: {test_acc*100:.2f}%")

        return model


    

    def get_logit_for_target_digit(self, x_flat):
        x = x_flat.reshape(self.un_flattened_shape)  # (H, W, 1)
        logits = self.cnn(x)
        top_class = jnp.argmax(logits)

        probs = logits / 3
        #probs = jax.nn.log_softmax(logits, axis=-1)

        top_is_not_target = (top_class - self.target_class) / (top_class - self.target_class + 1e-12)
        return probs[self.target_class] - probs[top_class]*top_is_not_target
  
    

    def reference_fn(self):
        return self.get_logit_for_target_digit





    def get_test_set(self, n_samples: int, order: int) -> DifferentialData:
        if n_samples > self.test_imgs.shape[0]:
            raise ValueError("Requested number of samples exceeds available test set size.")
        
        # return a subset of the test set
        indices = jnp.arange(self.test_imgs.shape[0])
        perm = random.permutation(self.key_test, indices)[:n_samples]

        x = self.test_imgs[perm]
        x_flat = x.reshape(n_samples, -1)  # (n_samples, n_dims)
        
        y_and_dy_fn = jax.value_and_grad(self.reference_fn())
        y, dydx= jax.vmap(y_and_dy_fn)(x_flat)
        #print("y shape: ", y.shape)
        #print("dy shape: ", dy.shape)

        ddy = None
        dddy = None
        if order >= 2:
            ddy = jax.vmap(jax.hessian(self.reference_fn()))(x_flat)
        if order >= 3:
            dddy = jax.vmap(jax.jacfwd(jax.hessian(self.reference_fn())))(x_flat) 
        
        return DifferentialData(
            order = order,
            x = x_flat,
            y = y,
            dy = dydx,
            ddy = ddy,
            dddy = dddy
        ) 
    


    def sample(self, key: PRNGKeyArray, n_samples: int, order: int = 1) -> DifferentialData:

        if n_samples > self.train_imgs.shape[0]:
            raise ValueError("Requested number of samples exceeds available train set size.")
        
        # return a subset of the train set
        indices = jnp.arange(self.train_imgs.shape[0])
        perm = random.permutation(key, indices)[:n_samples]

        x = self.train_imgs[perm]
        x_flat = x.reshape(n_samples, -1)  # flatten to (n_samples, n_dims)
        
        y_and_dy_fn = jax.value_and_grad(self.reference_fn())
        y, dydx= jax.vmap(y_and_dy_fn)(x_flat)  # y: (n_samples,), dy: (n_samples, n_dims)
        
        return DifferentialData(
            order = order,
            x = x_flat,
            y = y,
            dy = dydx
        )
    






    def plot_img_colored(self, img, figsize=(3,3), interpolation="nearest"):
    
        if len(img.shape) == 1:
            side_length = int(jnp.sqrt(img.shape[0]))
            arr = img.reshape(side_length, side_length)
        #else:
        #    arr = jnp.squeeze(img)



        # create a 3‐color colormap: red at low end, grey at zero, green at high end
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "red_grey_green", 
            [("red"), ("lightgrey"), ("green")],
            N=256
        )
        # center the color scaling at zero
        norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

        plt.figure(figsize=figsize)
        plt.imshow(arr, cmap=cmap, norm=norm, interpolation=interpolation)
        plt.axis("off")
        plt.show()






    def visualize_data(self, dataset: DifferentialData, name: str):
        # TODO
        print("shapes:")
        print("x shape: ", dataset.x.shape)
        print("y shape: ", dataset.y.shape)
        print("dydx shape: ", "-" if dataset.dy == None  else dataset.dy.shape)
        print("ddyddx shape: ", "-" if dataset.ddy == None  else dataset.ddy.shape)
        print("dddydddx shape: ", "-" if dataset.dddy == None  else dataset.dddy.shape)

        # for img 0 plot dy
        dy0 = dataset.dy[0]
        self.plot_img_colored(dy0)
        



