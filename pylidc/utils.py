import numpy as np
from .Annotation import Annotation

def consensus(anns, clevel=0.5, extent=None, verbose=True):
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

    extent: float, default=None
        If a float, the volumes will be padded so that they have
        physical dimensions >= `extent` units in each coordinate
        axis direction. The default (None) doesn't perform this
        operation.

    verbose: bool, default=True
        Turns the DICOM image loading message on/off.
    """
    if not all([isinstance(a, Annotation) for a in anns]):
        raise TypeError("`anns` should be list of `pylidc.Annotation`s.")

    clevel = float(clevel)

    if not (0.0 <= clevel <= 1.0):
        raise ValueError("`clevel` should be between 0 and 1.")

    scan = anns[0].scan
    rij = scan.pixel_spacing
    rk  = scan.slice_thickness

    # Load the images. Get the z positions.
    images = scan.load_all_dicom_images(verbose=verbose)
    img_zs = [float(img.ImagePositionPatient[-1]) for img in images]
    img_zs = np.unique(img_zs)

    bboxs = np.array([a.bbox(image_coords=1) for a in anns])

    # Last dimension is z dist, not index.
    for i in range(bboxs.shape[0]):
        # Float comparison ... probably not the best way.
        bboxs[i,2,0] = img_zs.searchsorted(bboxs[i,2,0])
        bboxs[i,2,1] = img_zs.searchsorted(bboxs[i,2,1])
    
    # Cast to int now that everyone is an index.
    bboxs = bboxs.astype(np.int)

    imin,jmin,kmin = np.array([b[:,0] for b in bboxs]).min(0)
    imax,jmax,kmax = np.array([b[:,1] for b in bboxs]).max(0)

    while (imax-imin)*rij < extent:
        imin -= 1 if imin > 0 else 0
        imax += 1 if imax < 511 else 0

    while (jmax-jmin)*rij < extent:
        jmin -= 1 if jmin > 0 else 0
        jmax += 1 if jmax < 511 else 0

    while (kmax-kmin)*rk  < extent:
        kmin -= 1 if kmin > 0 else 0
        kmax += 1 if kmax < (img_zs.shape[0]-1) else 0

    return_bbox = np.array([[imin, imax],
                            [jmin, jmax],
                            [kmin, kmax]])

    raise NotImplementedError()
