import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection)
import util
np.random.seed(1)


# TSNE cluster
def cluster(originData, target, feature_I, feature_V, pretrain_model_time_dir, iterations, is_tensorboard=True):
    print "start cluster",iterations
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=1)
    feature_tsne_V = tsne.fit_transform(feature_V)
    F_V_cluster = plot_embedding(feature_tsne_V, originData, target, pretrain_model_time_dir, title="Variables_"+str(iterations),is_tensorboard=is_tensorboard)
    feature_tsne_I = tsne.fit_transform(feature_I)
    F_I_cluster = plot_embedding(feature_tsne_I, originData, target, pretrain_model_time_dir, title="Identity_"+str(iterations),is_tensorboard=is_tensorboard)
    return F_I_cluster, F_V_cluster

# Scale and visualize the embedding vectors
def plot_embedding(X, originData, target, pretrain_model_time_dir, title=None, is_tensorboard=True):
    fig = plt.figure()
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    # plt.figure()
    ax =fig.add_subplot(111)
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
                offsetbox.OffsetImage(originData[i].reshape((28, 28)), zoom=0.5, cmap=plt.cm.gray_r), X[i])
            ax.add_artist(imagebox)

    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)
    if is_tensorboard:
        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        # print data.shape
        return data
    else:
        util.check_create_dir("./visualFig/")
        util.check_create_dir("./visualFig/{}/".format(pretrain_model_time_dir))
        fig.savefig("./visualFig/{}/visual{}.png".format(pretrain_model_time_dir, title))
