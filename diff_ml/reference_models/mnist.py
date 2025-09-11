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
from utils import Range


# ---------- Model (same as before) ----------
class CNN(eqx.Module):
    
    convs: list
    linear: eqx.nn.Linear
    #n_dims: int
    #un_flattened_shape: tuple

    def __init__(self, input_size: int, depth: int, base_channels: int, num_classes: int, key):
        keys = random.split(key, depth + 1)
        convs = []
        in_ch = 1
    
        for i in range(depth):
            out_ch = base_channels * (2**i)
            convs.append(eqx.nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, key=keys[i]))
            in_ch = out_ch
    
        # compute final spatial size exactly
        size = input_size
        for _ in range(depth):
            size = (size - 1) // 2 + 1
        final_size = size
    
    
    
        self.convs = convs
        self.linear = eqx.nn.Linear(in_ch * final_size * final_size, num_classes, key=keys[-1])
        #self.input_size = input_size
        
    def __call__(self, x):
        # x: (H, W, 1)
        x = jnp.transpose(x, (2, 0, 1))          # → (C,H,W)
        for conv in self.convs:
            x = conv(x)
            x = jax.nn.silu(x)
        x = x.reshape(-1)                       # flatten
        return self.linear(x)



class MNIST_ref(eqx.Module):

    key: PRNGKeyArray
    
    train_imgs: Float[Array, "train_size H W 1"] 
    test_imgs: Float[Array, "test_size H W 1"]
    train_lbls: Float[Array, "train_size"]
    test_lbls: Float[Array, "test_size"]

    cnn: CNN
    n_dims: int
    un_flattened_shape: tuple
    scale: float = 1.0  # default scale for MNIST
    target_class: int = 9  # default target class for MNIST

    was_normalized: bool = False
    
    def __init__(self, key, scale, target_class=9):
        self.key = key
        self.scale = scale
        self.target_class = target_class
        input_size = int(28 * scale)
        self.n_dims = input_size * input_size
        self.un_flattened_shape = (input_size, input_size, 1)

        self.key, cnn_key = random.split(self.key)
        self.cnn = CNN(
            input_size=int(28 * scale),
            depth=3,  # number of convolutional layers
            base_channels=32, # base number of channels in first conv layer
            num_classes=10,  # MNIST has 10 classes
            key=cnn_key
        )
        #self.cnn = CNN_downsampled(
        #    input_size=int(28 * scale),
        #    depth=3,
        #    base_channels=32,
        #    num_classes=10,
        #    key=cnn_key,
        #    anti_alias="avg",     
        #    blur_size=3            
        #)


        # Train the CNN on MNIST
        self.cnn = self.train(
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

        #self.train_imgs = train_imgs
        #self.train_lbls = train_lbls
        #self.test_imgs  = test_imgs
        #self.test_lbls  = test_lbls
        return train_imgs, train_lbls, test_imgs, test_lbls

    #def loss_fn(self, model, xb, yb):
    #    logits = jax.vmap(model)(xb)
    #    onehot = jax.nn.one_hot(yb, logits.shape[-1])
    #    return optax.softmax_cross_entropy(logits, onehot).mean()
    
    # smoothed labels
    def loss_fn(self, model, xb, yb, eps=0.1):
        logits = jax.vmap(model)(xb)
        num_classes = logits.shape[-1]
        soft = jax.nn.one_hot(yb, num_classes) * (1 - eps) + eps / num_classes
        return optax.softmax_cross_entropy(logits, soft).mean()



    def accuracy(self, model, xb, yb):
        preds = jnp.argmax(jax.vmap(model)(xb), axis=-1)
        return jnp.mean(preds == yb)
      # TODO smooth loss



    def train(self, lr, batch_size, epochs):

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
            self.key, perm_key = random.split(self.key)
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


    

    def ref_fn(self, x_flat):
        x = x_flat.reshape(self.un_flattened_shape)  # reshape to (H, W, 1)
        logits = self.cnn(x)
        probs = logits / 3.0
        #probs = jax.nn.log_softmax(logits, axis=-1)
        return probs[self.target_class]
    
    #def ref_fn(self, x_flat, T: float = 2.0):
    #    x = x_flat.reshape(self.un_flattened_shape)
    #    logits = self.cnn(x)
    #    probs = jax.nn.softmax(logits / T, axis=-1)
    #    return probs[self.target_class]

    

    def reference_fn(self, *args):
        return self.ref_fn





    def get_test_set(self, n_samples):
        if n_samples > self.test_imgs.shape[0]:
            raise ValueError("Requested number of samples exceeds available test set size.")
        # return a subset of the test set
        indices = jnp.arange(self.test_imgs.shape[0])
        perm = random.permutation(self.key, indices)[:n_samples]

        x = self.test_imgs[perm]
        x_flat = x.reshape(n_samples, -1)  # flatten to (n_samples, n_dims)
        
        y_and_dy_fn = jax.value_and_grad(self.ref_fn)
        y, dydx= jax.vmap(y_and_dy_fn)(x_flat)  # y: (n_samples,), dy: (n_samples, n_dims)
        #print("y shape: ", y.shape)
        #print("dy shape: ", dy.shape)

        ddyddx = jax.vmap(jax.hessian(self.ref_fn))(x_flat)
        #print("H_full shape: ", H_full.shape)
        
        return x_flat, y, dydx, ddyddx  
    


    def sample(self, key, n_samples):

        if n_samples > self.train_imgs.shape[0]:
            raise ValueError("Requested number of samples exceeds available train set size.")
        
        # return a subset of the train set
        indices = jnp.arange(self.train_imgs.shape[0])
        perm = random.permutation(key, indices)[:n_samples]

        x = self.train_imgs[perm]
        x_flat = x.reshape(n_samples, -1)  # flatten to (n_samples, n_dims)
        
        y_and_dy_fn = jax.value_and_grad(self.ref_fn)
        y, dydx= jax.vmap(y_and_dy_fn)(x_flat)  # y: (n_samples,), dy: (n_samples, n_dims)
        
        return x_flat, y, dydx, None
    

