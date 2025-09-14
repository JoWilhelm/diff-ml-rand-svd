import datetime
import matplotlib.pyplot as plt
import jax.numpy as jnp

import diff_ml.losses.regression as losses
from diff_ml.typing import DifferentialData



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



def plot_3d_differential_data(dataset: DifferentialData, name: str, x1_index=0, x2_index=1, x1_name="x1", x2_name="x2"):
    # visulaize the test set
    print("shapes:")
    print("x shape: ", dataset.x.shape)
    print("y shape: ", dataset.y.shape)
    print("dydx shape: ", "-" if dataset.dy == None  else dataset.dy.shape)
    print("ddyddx shape: ", "-" if dataset.ddy == None  else dataset.ddy.shape)
    print("dddydddx shape: ", "-" if dataset.dddy == None  else dataset.dddy.shape)

    # plot only over first two input dimensions
    xs = dataset.x[..., x1_index]
    ys = dataset.x[..., x2_index]

    # value
    plot_3d_data(xs, ys, dataset.y, x1_label=x1_name, x2_label=x2_name, y_label="y", title=f"{name} target\ny")

    # 1st order
    if dataset.order >= 1:
        plot_3d_data(xs, ys, dataset.dy[:, 0], x1_label=x1_name, x2_label=x2_name, y_label="dydx1", title=f"{name}\ndydx1")
        plot_3d_data(xs, ys, dataset.dy[:, 1], x1_label=x1_name, x2_label=x2_name, y_label="dydx1", title=f"{name}\ndydx2")

    # 2nd order
    if dataset.order >= 2:
        plot_3d_data(xs, ys, dataset.ddy[:, 0, 0], x1_label=x1_name, x2_label=x2_name, y_label="ddyddx11", title=f"{name}\nddyddx11")
        plot_3d_data(xs, ys, dataset.ddy[:, 0, 1], x1_label=x1_name, x2_label=x2_name, y_label="ddyddx12", title=f"{name}\nddyddx12")
        plot_3d_data(xs, ys, dataset.ddy[:, 1, 0], x1_label=x1_name, x2_label=x2_name, y_label="ddyddx21", title=f"{name}\nddyddx21")
        plot_3d_data(xs, ys, dataset.ddy[:, 1, 1], x1_label=x1_name, x2_label=x2_name, y_label="ddyddx22", title=f"{name}\nddyddx22")

    # 3rd order
    if dataset.order >= 3:
        plot_3d_data(xs, ys, dataset.dddy[:, 0, 0, 0], x1_label=x1_name, x2_label=x2_name, y_label="dddydddx111", title=f"{name}\ndddydddx111")
        plot_3d_data(xs, ys, dataset.dddy[:, 0, 0, 1], x1_label=x1_name, x2_label=x2_name, y_label="dddydddx112", title=f"{name}\ndddydddx112")
        plot_3d_data(xs, ys, dataset.dddy[:, 0, 1, 0], x1_label=x1_name, x2_label=x2_name, y_label="dddydddx121", title=f"{name}\ndddydddx121")
        plot_3d_data(xs, ys, dataset.dddy[:, 0, 1, 1], x1_label=x1_name, x2_label=x2_name, y_label="dddydddx122", title=f"{name}\ndddydddx122")
        plot_3d_data(xs, ys, dataset.dddy[:, 1, 0, 0], x1_label=x1_name, x2_label=x2_name, y_label="dddydddx211", title=f"{name}\ndddydddx211")
        plot_3d_data(xs, ys, dataset.dddy[:, 1, 0, 1], x1_label=x1_name, x2_label=x2_name, y_label="dddydddx212", title=f"{name}\ndddydddx212")
        plot_3d_data(xs, ys, dataset.dddy[:, 1, 1, 0], x1_label=x1_name, x2_label=x2_name, y_label="dddydddx221", title=f"{name}\ndddydddx221")
        plot_3d_data(xs, ys, dataset.dddy[:, 1, 1, 1], x1_label=x1_name, x2_label=x2_name, y_label="dddydddx222", title=f"{name}\ndddydddx222")


