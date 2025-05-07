import datetime
import matplotlib.pyplot as plt




# visualize generated data before trianig 

def plot_generated_data():
    vis_dim = 0
    fig, axs = plt.subplots(1, 2)

    plot_payoff_data(axs[0], X[:,vis_dim], Y[:,vis_dim], baskets[:, vis_dim], prices[:,vis_dim])
    plot_delta_data(axs[1], X[:,vis_dim], baskets[:, vis_dim], Z[:,vis_dim], deltas[:,vis_dim])

    plt.show()





# visualoze model predictions

def plot_eval(model, pred: Predictions, test_set: TestSet):

    def print_rmse(pred, true):
        plt.title(f"RMSE: {rmse(true, pred)}")

    x_test, baskets, y_test, dydx_test, vegas, gammas = astuple(test_set)
    pred_y, pred_dydx, pred_ddyddx = astuple(pred)
    pred_y = pred_y[:, jnp.newaxis]

    # Create a single figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot the first subplot
    axes[0].plot(baskets, pred_y, '.', markersize=1)
    axes[0].plot(baskets, y_test, '.', markersize=1)
    axes[0].legend(['Pred Price', 'True Price'], loc='upper left')
    axes[0].set_title(f"Values \n {rmse(pred_y, y_test)}")

    # Plot the second subplot
    dydx_idx = 0
    axes[1].plot(baskets, pred_dydx[:, dydx_idx], '.', markersize=1)
    axes[1].plot(baskets, dydx_test[:, dydx_idx], '.', markersize=1)
    axes[1].legend(['Pred Delta', 'True Delta'], loc='upper left')
    axes[1].set_title(f"Differentials\n{rmse(pred_dydx, dydx_test)}")

    # Calculate and plot gammas in the third subplot
    pred_gammas = jnp.sum(pred_ddyddx, axis=(1, 2))
    axes[2].plot(baskets, pred_gammas, '.', markersize=1, label='Pred')
    axes[2].plot(baskets, gammas, '.', markersize=1, label='True')
    axes[2].legend()
    axes[2].set_title(f"Gammas\n{rmse(pred_gammas, gammas)}")

    # Adjust the layout and save the figure to a PDF file
    plt.tight_layout()
    plt.show()
    now = datetime.datetime.now()
    fig.savefig(f'results/all_at_once/eval_ml_{now}.pdf', bbox_inches='tight')

plot_eval(model, pred, test_set)
