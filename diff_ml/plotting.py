from diff_ml.typing import DifferentialData
import matplotlib.pyplot as plt




def plot_3d_data(x1, x2, y, x1_label, x2_label, y_label, title=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d') 
    sc = ax.scatter(x1, x2, y, c=y)

    # Add a colorbar to show the mapping of colors to z-values
    cbar = fig.colorbar(sc)
    cbar.set_label(y_label)
    
    ax.set_title(title if title else '3D Scatter Plot')
    ax.set_xlabel(x1_label)
    ax.set_ylabel(x2_label)
    ax.set_zlabel(y_label) # type: ignore
    return fig



def plot_3d_differential_data(dataset: DifferentialData, name: str, x1s=None, x2s=None, x1_index=0, x2_index=1, x1_name="x1", x2_name="x2"):
    # visulaize the test set
    print("shapes:")
    print("x shape: ", dataset.x.shape)
    print("y shape: ", dataset.y.shape)
    print("dydx shape: ", "-" if dataset.dy == None  else dataset.dy.shape)
    print("ddyddx shape: ", "-" if dataset.ddy == None  else dataset.ddy.shape)
    print("dddydddx shape: ", "-" if dataset.dddy == None  else dataset.dddy.shape)

    if x1s is None or x2s is None:
        # plot only over given input dimensions
        x1s = dataset.x[..., x1_index]
        x2s = dataset.x[..., x2_index]

    # value
    plot_3d_data(x1s, x2s, dataset.y, x1_label=x1_name, x2_label=x2_name, y_label="y", title=f"{name} target\ny")

    # 1st order
    if dataset.order >= 1:
        plot_3d_data(x1s, x2s, dataset.dy[:, 0], x1_label=x1_name, x2_label=x2_name, y_label="dydx1", title=f"{name}\ndydx1")
        plot_3d_data(x1s, x2s, dataset.dy[:, 1], x1_label=x1_name, x2_label=x2_name, y_label="dydx1", title=f"{name}\ndydx2")

    # 2nd order
    if dataset.order >= 2 and dataset.ddy is not None:
        plot_3d_data(x1s, x2s, dataset.ddy[:, 0, 0], x1_label=x1_name, x2_label=x2_name, y_label="ddyddx11", title=f"{name}\nddyddx11")
        plot_3d_data(x1s, x2s, dataset.ddy[:, 0, 1], x1_label=x1_name, x2_label=x2_name, y_label="ddyddx12", title=f"{name}\nddyddx12")
        plot_3d_data(x1s, x2s, dataset.ddy[:, 1, 0], x1_label=x1_name, x2_label=x2_name, y_label="ddyddx21", title=f"{name}\nddyddx21")
        plot_3d_data(x1s, x2s, dataset.ddy[:, 1, 1], x1_label=x1_name, x2_label=x2_name, y_label="ddyddx22", title=f"{name}\nddyddx22")

    # 3rd order
    if dataset.order >= 3 and dataset.dddy is not None:
        plot_3d_data(x1s, x2s, dataset.dddy[:, 0, 0, 0], x1_label=x1_name, x2_label=x2_name, y_label="dddydddx111", title=f"{name}\ndddydddx111")
        plot_3d_data(x1s, x2s, dataset.dddy[:, 0, 0, 1], x1_label=x1_name, x2_label=x2_name, y_label="dddydddx112", title=f"{name}\ndddydddx112")
        plot_3d_data(x1s, x2s, dataset.dddy[:, 0, 1, 0], x1_label=x1_name, x2_label=x2_name, y_label="dddydddx121", title=f"{name}\ndddydddx121")
        plot_3d_data(x1s, x2s, dataset.dddy[:, 0, 1, 1], x1_label=x1_name, x2_label=x2_name, y_label="dddydddx122", title=f"{name}\ndddydddx122")
        plot_3d_data(x1s, x2s, dataset.dddy[:, 1, 0, 0], x1_label=x1_name, x2_label=x2_name, y_label="dddydddx211", title=f"{name}\ndddydddx211")
        plot_3d_data(x1s, x2s, dataset.dddy[:, 1, 0, 1], x1_label=x1_name, x2_label=x2_name, y_label="dddydddx212", title=f"{name}\ndddydddx212")
        plot_3d_data(x1s, x2s, dataset.dddy[:, 1, 1, 0], x1_label=x1_name, x2_label=x2_name, y_label="dddydddx221", title=f"{name}\ndddydddx221")
        plot_3d_data(x1s, x2s, dataset.dddy[:, 1, 1, 1], x1_label=x1_name, x2_label=x2_name, y_label="dddydddx222", title=f"{name}\ndddydddx222")


