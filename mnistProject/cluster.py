import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection)
np.random.seed(1)


# TSNE cluster
def cluster(originData, target, feature_I, feature_V):
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=1)
    feature_tsne_V = tsne.fit_transform(feature_V)
    plot_embedding(feature_tsne_V, originData, target, title="Variables")
    feature_tsne_I = tsne.fit_transform(feature_I)
    plot_embedding(feature_tsne_I, originData, target, title="Identity")


# Scale and visualize the embedding vectors
def plot_embedding(X, originData, target, title=None):
    plt.figure()
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    # plt.figure()
    ax = plt.subplot(111)

    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(target[i]),
                 color=plt.cm.Set1(target[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[2., 2.]])  # just something big
        for i in range(originData.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 1e-2:
                # don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(originData[i].reshape((28, 28)), zoom=0.5, cmap=plt.cm.gray_r),
                X[i])
            ax.add_artist(imagebox)

    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)
    plt.savefig("./visualFig/visual{}.png".format(title))
