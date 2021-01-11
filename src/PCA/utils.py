import numpy as np
import matplotlib.pyplot as plt

def standardizing(data, axis=0) :
    mean = np.mean(data, axis=axis)
    std = np.std(data, axis=axis)
    # handle zero in scale (all values the same)
    std[std == 0.0] = 1.0
    return (data - mean) / std, mean, std

def makeGrid(array, nrows=3):
    nindex, height, width = array.shape
    ncols = nindex//nrows
    assert nindex == nrows*ncols
    return array.reshape(nrows, ncols, height, width, -1).swapaxes(1,2).reshape(height*nrows, width*ncols, -1)

def displayRowComparaison(data, dataDescription = None, labels = None, annotations = None, byRowsAnnotations = False, cleanAxis = True, hAxesPad=0., vAxesPad = 0.):
    rows, cols = data.shape[:2]

    for i in range(1, rows):
        assert data[i-1].shape == data[i].shape, 'all data must have the same shape'

    assert dataDescription is None or rows == len(dataDescription), 'dataDescription must match the number of data'
    assert labels is None or cols == len(labels), 'labels must match the number of labels in data'

    if byRowsAnnotations:
        assert annotations is None or annotations.shape == data.shape[:2], 'annotations shape must match data shape'
    else: 
        assert annotations is None or len(labels) == len(annotations), 'annotations size must match labels size'

    fig, axs = plt.subplots(rows, cols, figsize=(cols*2, rows*2+1), facecolor='w', edgecolor='k')
    axs = axs.reshape(rows, cols) # fix 1D array
    fig.subplots_adjust(hspace=vAxesPad, wspace=hAxesPad)

    for ax, img in zip(axs.ravel(), data.reshape(-1, *data.shape[-2:])): # display images
        ax.imshow(img, cmap='gray')
    
    if labels is not None:
        for ax, l in zip(axs[0], labels): # add target titles
            ax.set_title(l)
    
    if cleanAxis is not None: # clean axis
        for ax in axs.ravel():
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

    if dataDescription is not None:
        for ax, desc in zip(axs[:, 0], dataDescription):
            ax.get_yaxis().set_visible(True)
            if cleanAxis:
                ax.set_yticklabels([])
            ax.set_ylabel(desc, rotation=90, size='large')

    if annotations is not None:
        for ax, annotation in zip(axs[-1], annotations) if not byRowsAnnotations else zip(axs.ravel(), annotations.ravel()):
            ax.get_xaxis().set_visible(True)
            if cleanAxis:
                ax.set_xticklabels([])
            ax.set_xlabel(annotation)
    plt.show()
    return fig
