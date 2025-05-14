import datetime
import matplotlib.pyplot as plt
import jax.numpy as jnp

import diff_ml.losses.regression as losses

## visualize generated data before trianig 
#
#def plot_generated_data():
#    vis_dim = 0
#    fig, axs = plt.subplots(1, 2)
#
#    plot_payoff_data(axs[0], X[:,vis_dim], Y[:,vis_dim], baskets[:, vis_dim], prices[:,vis_dim])
#    plot_delta_data(axs[1], X[:,vis_dim], baskets[:, vis_dim], Z[:,vis_dim], deltas[:,vis_dim])
#
#    plt.show()




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
