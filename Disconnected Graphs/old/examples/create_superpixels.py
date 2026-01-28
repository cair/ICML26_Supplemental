from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import slic


def plot_graph_from_image(image, desired_nodes=75, save_in=None):
    segments = slic(image, start_label=0, slic_zero=True)

    # show the output of SLIC
    fig = plt.figure("Superpixels")
    ax = fig.add_subplot(1, 1, 1)
    # ax.imshow(mark_boundaries(image, segments), cmap="gray")
    ax.imshow(image)  # , cmap="gray")
    plt.axis("off")

    asegments = np.array(segments)

    # From https://stackoverflow.com/questions/26237580/skimage-slic-getting-neighbouring-segments

    segments_ids = np.unique(segments)

    # centers
    centers = np.array(
        [np.mean(np.nonzero(segments == i), axis=1) for i in segments_ids]
    )

    vs_right = np.vstack([segments[:, :-1].ravel(), segments[:, 1:].ravel()])
    vs_below = np.vstack([segments[:-1, :].ravel(), segments[1:, :].ravel()])
    bneighbors = np.unique(np.hstack([vs_right, vs_below]), axis=1)

    plt.scatter(centers[:, 1], centers[:, 0], c="r")

    breakpoint()

    for i in range(bneighbors.shape[1]):
        y0, x0 = centers[bneighbors[0, i]]
        y1, x1 = centers[bneighbors[1, i]]

        l = Line2D([x0, x1], [y0, y1], c="r", alpha=0.5)
        ax.add_line(l)

    # show the plots
    if save_in is None:
        plt.show()
    else:
        plt.savefig(save_in, bbox_inches="tight")
    plt.close()


def gen_superpixels(img, n_nodes=75):
    labels = slic(img, n_nodes, slic_zero=True, channel_axis=None)
    breakpoint()


def main():

    # from keras.api.datasets import mnist
    # (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    # # X_train = np.where(X_train > 75, 1, 0)
    # # X_test = np.where(X_test > 75, 1, 0)
    # Y_train = Y_train.astype(np.uint32)
    # Y_test = Y_test.astype(np.uint32)
    #
    # plot_graph_from_image(X_train[0], desired_nodes=75)
    # gen_superpixels(X_train[0])
    from torchvision.datasets import MNIST
    print("Reading dataset")
    dset = MNIST("./data",download=True)
    imgs = dset.data.unsqueeze(-1).numpy().astype(np.float64)
    labels = dset.targets.numpy()

    breakpoint()


if __name__ == "__main__":
    main()
