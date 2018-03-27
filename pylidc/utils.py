import numpy as np
from .Annotation import Annotation

def consensus(anns, clevel=0.5, pad=None, ret_masks=True, verbose=True):
    """
    Return the boolean-valued consensus volume amongst the
    provided annotations (`anns`) at a particular consensus level
    (`clevel`).

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

    returns: consensus_mask, consensus_bbox[, masks]
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
