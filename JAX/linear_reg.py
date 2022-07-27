import pandas as pd
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import grad, vmap, jit
from jax import random
from sklearn.datasets import load_iris, load_diabetes


DARK_BLUE: str = '#1F77B4'


def laod_scikit_data(task: str = 'regression'):
    """Load dataset for the preferred task"""
    if task == 'classification':
        return load_iris(return_X_y=True, as_frame=True)
    elif task == 'regression':
        return load_diabetes(return_X_y=True, as_frame=True)


def plot_model(Y, Y_hat, w):
    fig = plt.figure(figsize=(10,8))
    plt.scatter(range(Y.shape[0]), Y)
    plt.scatter(range(Y.shape[0]), Y_hat)
    plt.legend(['True Values', 'Predictions'])
    plt.title('Linear Regression Model Prediction')
    plt.show()


def random_key_init(key: int =42):
    """Initializing random key for JAX library"""
    rnd_key = random.PRNGKey(key)
    return rnd_key


def add_bias_into_matrices(X):
    """Add initial column of ones to act as bias"""
    return jnp.insert(X, 0, 1, axis=1)


def linear_reg_pred(X, w):
    """Making prediction for each sample in the data"""
    return jnp.dot(X, w)


def mse_loss(X, Y, w):
    """Calculating the MSE loss using true values and prediction"""
    Y_hat = linear_reg_pred(X, w)
    return jnp.sum(Y - Y_hat)/Y.shape[0]


def linear_reg(X, Y):
    """Train linear regression model for given X and Y"""
    X_np = add_bias_into_matrices(X.values)  # Convert pd.DataFrame into np array
    # Convert labels to np array and add one more dim to mach sizes
    Y_np = jnp.expand_dims(Y.values, -1)
    w = jnp.dot(jnp.linalg.inv(jnp.dot(X_np.T,X_np)),jnp.dot(X_np.T,Y_np))
    mse = mse_loss(X_np, Y_np, w)  # Calculate the loss
    print('Linear Reg MSE error :', mse)
    Y_hat_linear_reg = linear_reg_pred(X_np, w)
    plot_model(Y, Y_hat_linear_reg, w)
    return Y_hat_linear_reg, w


if __name__ == '__main__':
    task = 'regression'
    X, Y = laod_scikit_data(task) 
    linear_reg(X, Y)
