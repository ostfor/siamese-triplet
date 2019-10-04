import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


def imscatter(x, y, data, embs, ax=None, zoom=1):
    if ax is None:
        ax = plt.gca()
    x, y = np.atleast_1d(x, y)
    artists = []
    for image, coord in zip(data, embs):
        im = OffsetImage(image, zoom=zoom)
        ab = AnnotationBbox(im, coord, xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))

    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists


# x = np.linspace(0, 10, 20)
# y = np.cos(x)
def plot_embeddings(data, embs):
    fig, ax = plt.subplots()
    fig.set_size_inches(18.5, 10.5)

    print(embs.tolist())
    x, y = list(zip(*embs.tolist()))
    print(x, y)
    imscatter(x, y, data, embs, zoom=1, ax=ax)

    ax.plot(x, y, color='white')
    plt.show()