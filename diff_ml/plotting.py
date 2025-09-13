import datetime
import matplotlib.pyplot as plt
import jax.numpy as jnp

import diff_ml.losses.regression as losses




# visualIze model predictions

def plot_eval(pred_y, pred_dydx, pred_ddyddx, test_ds):


    baskets = test_ds["baskets"]
    y_test = test_ds["y"]
    dydx_test = test_ds["dydx"]
    gammas = test_ds["ddyddx"]
    
    pred_y = pred_y[:, jnp.newaxis]

    # Create a single figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot the first subplot
    axes[0].plot(baskets, pred_y, '.', markersize=1)
    axes[0].plot(baskets, y_test, '.', markersize=1)
    axes[0].legend(['Pred Price', 'True Price'], loc='upper left')
    axes[0].set_title(f"Values \n rmse: {losses.rmse(pred_y, y_test)}")

    # Plot the second subplot
    dydx_idx = 0
    axes[1].plot(baskets, pred_dydx[:, dydx_idx], '.', markersize=1)
    axes[1].plot(baskets, dydx_test[:, dydx_idx], '.', markersize=1)
    axes[1].legend(['Pred Delta', 'True Delta'], loc='upper left')
    axes[1].set_title(f"Differentials\nrmse: {losses.rmse(pred_dydx, dydx_test)}")

    # Calculate and plot gammas in the third subplot
    pred_gammas = jnp.sum(pred_ddyddx, axis=(1, 2))
    axes[2].plot(baskets, pred_gammas, '.', markersize=1, label='Pred')
    axes[2].plot(baskets, gammas, '.', markersize=1, label='True')
    axes[2].legend()
    axes[2].set_title(f"Gammas\nrmse: {losses.rmse(pred_gammas, gammas)}")

    # Adjust the layout and save the figure to a PDF file
    plt.tight_layout()
    #plt.show()
    now = datetime.datetime.now()
    fig.savefig(f'result/eval_ml_{now}.pdf', bbox_inches='tight')






def plot_3d_data(x1, x2, y, x1_label, x2_label, y_label, title=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(x1, x2, y, c=y, cmap=plt.cm.viridis)

    # Add a colorbar to show the mapping of colors to z-values
    cbar = fig.colorbar(sc)
    cbar.set_label(y_label)
    
    ax.set_title(title if title else '3D Scatter Plot')
    ax.set_xlabel(x1_label)
    ax.set_ylabel(x2_label)
    ax.set_zlabel(y_label)
    return fig
