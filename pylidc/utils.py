import numpy as np
from .Annotation import Annotation
import matplotlib.pyplot as plt
from skimage.measure import find_contours
from matplotlib.widgets import Slider

def consensus(anns, clevel=0.5, pad=None, ret_masks=True, verbose=True):
    """Return the boolean-valued consensus volume amongst the
    provided annotations (`anns`) at a particular consensus level
    (`clevel`).

    Parameters
    ----------
    anns: list of `pylidc.Annotation` objects
        This list should be probably be one of the lists
        returned by the `pylidc.Scan.cluster_annotations`
        routine.

    clevel: float, default=0.5
        The consensus fraction. For example, if clevel=0.5, then
        a voxel will have value 1 in the returned boolean volume 
        when >= 50% of the segmentations include that voxel, and 0
        otherwise.

    pad: int, list, or float, default=None
        See `Annotation.bbox` for description for this argument.

    ret_masks: bool, default=True
        If True, a list of masks is also returned corresponding to
        all the annotations. Note that this slightly different than calling
        `boolean_mask` on each respective Annotation object because these 
        volumes will be the same shape and in a common reference frame.

    verbose: bool, default=True
        Turns the DICOM image loading message on/off.

    Returns
    -------
    consensus_mask, consensus_bbox[, masks]: (ndarray, tuple[, list])
        `consensus_mask` is the boolean-valued volume of the annotation
        masks at `clevel` consensus. `consensus_bbox` is a 3-tuple of 
        slices that can be used to index into the image volume at the 
        corresponding location of `consensus_mask`. `masks` is a list of
        boolean-valued mask volumes corresponding to each Annotation object.
        Each mask in the `masks` list has the same shape and is sampled in 
        the common reference frame provided by `consensus_bbox`.
    """
    bmats = np.array([a.bbox_matrix(pad=pad) for a in anns])
    imin,jmin,kmin = bmats[:,:,0].min(axis=0)
    imax,jmax,kmax = bmats[:,:,1].max(axis=0)

    # consensus_bbox
    cbbox = np.array([[imin,imax],
                      [jmin,jmax],
                      [kmin,kmax]])

    masks = [a.boolean_mask(bbox=cbbox) for a in anns]
    cmask = np.mean(masks, axis=0) >= clevel
    cbbox = tuple(slice(cb[0], cb[1]+1, None) for cb in cbbox)

    if ret_masks:
        return cmask, cbbox, masks
    else:
        return cmask, cbbox

def volume_viewer(vol, mask=None, **line_kwargs):
    """
    Interactive volume viewer utility

    Parameters
    ----------
    vol: ndarray, ndim=3
        An image volume.

    mask: ndarray, ndim=3, dtype=bool
        A boolean mask volume.

    line_kwargs: args
        Any keyword arguments that can be passed to `matplotlib.pyplot.plot`.

    Example
    -------
    An example::

        import pylidc as pl
        from pylidc.utils import volume_viewer

        ann = pl.query(pl.Annotation).first()
        vol = ann.scan.to_volume()

        padding = 70.0

        mask = ann.boolean_mask(pad=padding)
        bbox = ann.bbox(pad=padding)

        volume_viewer(vol[bbox], mask, ls='-', lw=2, c='r')

    """
    assert vol.ndim
    if mask is not None:
        if mask.dtype != bool:
            raise TypeError("mask was not bool type.")
        if vol.shape != mask.shape:
            raise ValueError("Shape mismatch between image volume and mask.")

    k = int(0.5*vol.shape[2])

    fig,ax = plt.subplots()
    plt.subplots_adjust(left=0, bottom=0.25)
    img = ax.imshow(vol[:,:,k], vmin=vol.min(), 
                    vmax=vol.max(), cmap=plt.cm.gray)
    ax.axis('off')

    if mask is not None:
        contours = []
        for i in range(vol.shape[2]):
            contour = []
            for c in find_contours(mask[:,:,i].astype(np.float), 0.5):
                line = ax.plot(c[:,1], c[:,0], **line_kwargs)[0]
                line.set_visible(0)
                contour.append(line)
            contours.append(contour)

    axslice = plt.axes([0.1, 0.1, 0.75, 0.03], facecolor='0.8')
    sslice  = Slider(axslice, 'Slice', 0, vol.shape[2]-1,
                     valinit=k, valstep=1)
    
    def update(i):
        i = int(i)
        img.set_data(vol[:,:,i])
        if mask is not None:
            for ic,contour in enumerate(contours):
                for c in contours[ic]:
                    c.set_visible(ic == i)
        fig.canvas.draw_idle()
    sslice.on_changed(update)

    update(k)
    plt.show()
