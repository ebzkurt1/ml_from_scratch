import pandas as pd
import jax.numpy as jnp
from jax import grad, vmap, jit
from jax import random
from sklearn.datasets import load_iris, load_diabetes


def laod_scikit_data(task: str = 'regression') -> pd.DataFrame:
    if task == 'classification':
        return load_iris(return_X_y=True, as_frame=True)
    elif task == 'regression':
        return load_diabetes(return_X_y=True, as_frame=True)


def random_key_init(key: int =42) -> None:
    """Initializing random key for JAX library"""
    rnd_key = random.PRNGKey(key)
    return rnd_key


def linear_reg_pred(X, W, b):
    """Making prediction for each sample in the data"""
    return jnp.dot(X,W) + b


def mse_loss(X, Y, W, b):
    """Calculating the MSE loss using true values and prediction"""
    Y_hat = linear_reg_pred(X, W, b)
    return jnp.sum(Y - Y_hat)/Y.shape[0]


def linear_reg(X, Y, epochs=10, lr=0.1):
    key = random_key_init()  # Initialize random key for the JAX
    X_np = X.values  # Convert pd.DataFrame into np array
    Y_np = jnp.expand_dims(Y.values, -1)
    n, m = X_np.shape  # Dimensions of the data
    W = random.normal(key, (m, 1))  # Initialize random weights
    b = random.normal(key, (1, 1))  # Initialize bias
    mse = mse_loss(X_np, Y_np, W, b)  # Calculate the loss
    print('Base MSE error :', mse)
    for e in range(epochs):
        W -= lr * grad(mse_loss,argnums=2)(X_np, Y_np, W, b)
        b -= lr * grad(mse_loss,argnums=3)(X_np, Y_np, W, b)
        print('Epoch', e, ' ----> MSE:', mse_loss(X_np, Y_np, W, b))

if __name__ == '__main__':
    task = 'regression'
    X, Y = laod_scikit_data(task) 
    linear_reg(X, Y, epochs= 50, lr=1)
