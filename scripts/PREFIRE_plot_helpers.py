import numpy as np
import matplotlib.pyplot as plt

def _scatterhex_hist_axes(fignum=5):
    """
    helper for scatterhex_hist to set up the axes with a nice
    setup. This has some hardcoded values that look nice for
    the (8,8) size figure.
    """
    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.15, 0.60
    spacing = 0.005
    cb_bottom = 0.06
    cb_height = 0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom, 0.2, height]
    rect_cb = [left, cb_bottom, width, cb_height]

    # start with a square Figure
    fig = plt.figure(fignum, figsize=(8, 8))
    fig.clf()

    ax = fig.add_axes(rect_scatter)
    ax_histx = fig.add_axes(rect_histx, sharex=ax)
    ax_histy = fig.add_axes(rect_histy, sharey=ax)
    ax_cb = fig.add_axes(rect_cb)

    return fig, (ax, ax_histx, ax_histy, ax_cb)


def scatterhex_hist(x, y, logx=False, logy=False, fignum=5,
                    xbins=100, ybins=100,
                    **hexbinkw):
    """
    a compound plot that uses hexbin to display a 2D histogram
    of x and y, and shows the marginal distributions (the x-only
    and y-only 1-D histograms) along the top and right sides.

    Parameters:
    -----------

    x : ndarray
        the x-coordinate data
    y : ndarray
        the y-coordinate data
    xbins : integer or array_like
        set to matplotlib.hist() for the x histogram
    ybins : integer or array_like
        set to matplotlib.hist() for the y histogram
    logx : boolean
        specifies if the x-coordinate should be plotted in log10 space.
    logy : boolean
        specifies if the y-coordinate should be plotted in log10 space.
    fignum : integer
        figure number to create

    any extra keywords are sent to hexbin().
    
    Returns:
    --------

    fig : Figure
        the created matplotlib figure object.
    axs : tuple
        a tuple containing the created matplotlib graphics objects:
        (hexbin axis, 1D x-axis along top, 1D y-axis along right,
         colorbar axis, colorbar)
        These can be used for customizations.
        
    """

    fig, (ax, ax_histx, ax_histy, ax_cb) = _scatterhex_hist_axes()

    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    if logx:
        xplot = np.log10(x)
    else:
        xplot = x

    if logy:
        yplot = np.log10(y)
    else:
        yplot = y

    # the scatter plot:
    hb = ax.hexbin(xplot, yplot, **hexbinkw)
    cb = plt.colorbar(hb, cax=ax_cb, orientation='horizontal')

    # determining if bins='log' was set to hexbin is tricky
    # because it can take several different types of input.
    if 'bins' in hexbinkw:
        if type(hexbinkw['bins']) is str:
            logbins = hexbinkw['bins'] == 'log'
        else:
            logbins = False
    else:
        logbins = False

    if logbins:
        cb.set_label('log10 counts')
    else:
        cb.set_label('counts')

    ax_histx.hist(xplot, bins=xbins)
    ax_histy.hist(yplot, bins=ybins, orientation='horizontal')

    return fig, (ax, ax_histx, ax_histy, ax_cb, cb)


def stripe_hist(x_, **kwargs):
    """
    Compute a "striped" histogram (for lack of a better name...)
    The idea is to compute the histogram for the values in each row of 
    the two-dimensional array x; the basic function np.histogram computes 
    a histogram over the flattened array. Instead, the histogram is 
    computed row-wise.
    Specifically, for x with shape [N,K], and bin edges with shape [B+1,], 
    the result is a [N,B] histogram, where each row [n,:] represents 
    the histogram of the elements [n,:] of x.

    input:
    x, 2D data array, shaped [N,K]

    remaining keyword arguments are sent to np.histogram.

    returns:
    xhist, a 2D array with shape [N,B]
    bin_edges, a 1-D array with shape [B+1,]
    """

    # method: compute a histogram of the first row, and then check the 
    # shape of the returned array. Then, we can create the proper shape 
    # for xhist. This allows us to skip attempting to calculate the 
    # shape from the kwargs.

    x = np.asarray(x_)

    if x.ndim != 2:
        raise ValueError('x must be a 2-dimensional array.')

    hist_0, bin_edges = np.histogram(x[0,:], **kwargs)

    if 'bins' in kwargs:
        del kwargs['bins']

    xhist = np.zeros((x.shape[0], hist_0.shape[0]))
    xhist[0,:] = hist_0

    for n in range(1, x.shape[0]):
        hist_n, _ = np.histogram(x[n,:], bins=bin_edges, **kwargs)
        xhist[n,:] = hist_n

    return xhist, bin_edges
