"""
Assorted helper functions which don't really fit in any of the 
main classes because that are too specific for a particular 
application, lack documentation, lack robustness, or for other reasons.
"""
import os
import numpy as np
from scipy.interpolate import RegularGridInterpolator

def interp_uniform(ann, side_length=None, verbose=True):
    """
    Get volumes corresponding to the CT values about the nodule 
    and its corresponding boolean indicator mask. The resulting 
    volume will be shape `side_length+1` along each dimension.
    The voxel-spacing for these volumes will be a uniform, 1mm.

    If the `side_length` is given such that the volume requires 
    information outside of the image bounds, these values are
    set to be the minimum value of the CT volume.

    side_length: integer, default None
        The physical length of each side of the new cubic 
        volume in millimeters. The default, `None`, takes the
        max of the nodule's bounding box dimensions.

        If this parameter is not `None`, then it should be 
        greater than any bounding box dimension, and in 
        this case, the volumes are expanded to exactly the 
        `side_length` specified if `new_length` divides 
        `side_length` evenly. If `new_spacing` does not 
        divide `side_length` evenly, then the resulting 
        volume dimensions will be less than `side_length`.
        In particular, `floor(side_length/new_spacing)`
        is used.

        If the specified `side_length` requires a padding 
        which results in an out-of-bounds image index, 
        then the image is padded with a constant value.

    verbose: boolean, default False
        Turn the loading statement on / off.

    returns: nodule, mask
        `nodule` and `mask` are the resampled CT, and boolean 
        volumes, respectively.

    Example: TODO
    """
    bbox  = ann.bbox(image_coords=True)
    bboxd = ann.bbox_dimensions(image_coords=True)
    rxy   = ann.scan.pixel_spacing

    # Begin input checks.
    if side_length is None:
        side_length = np.ceil(bboxd.max())
    else:
        if side_length < bboxd.max():
            raise ValueError('`side_length` must be greater\
                               than any bounding box dimension.')
    side_length = float(side_length)
    # End input checks.

    # Load the images. Get the z positions.
    images = ann.scan.load_all_dicom_images(verbose=verbose)
    img_zs = np.unique([
        float(img.ImagePositionPatient[-1]) for img in images
    ])

    # Initialize the bounding box and mask.
    mask = ann.get_boolean_mask()

    # Get the z values of the contours.
    contour_zs = np.unique([
        cnt.image_z_position for cnt in ann.contours
    ])

    zi_start = (np.abs(bbox[2,0]-img_zs)).argmin()
    zi_stop  = (np.abs(bbox[2,1]-img_zs)).argmin()

    ########################################################
    # { Begin mask corrections.

    # This conditional block handles the case where 
    # the contour annotations "skip a slice".
    if mask.shape[2] != (zi_stop-zi_start+1):
        old_mask = mask.copy()
        
        # Create the new mask with appropriate z-length.
        mask = np.zeros((old_mask.shape[0],
                         old_mask.shape[1],
                         zi_stop-zi_start+1), dtype=np.bool)

        # Map z's to an integer.
        z_to_index = dict(zip(
                        img_zs[zi_start:zi_stop+1],
                        range(img_zs[zi_start:zi_stop+1].shape[0])
                     ))

        # Map each slice to its correct location.
        for k in range(old_mask.shape[2]):
            mask[:,:, z_to_index[contour_zs[k]]] = old_mask[:,:,k]

    # } End mask corrections.
    ########################################################

    # These are upper / lower bounds for the resampling grid.
    # `*_max - *_min = side_length` by construction.
    xhat_min = 0.5*(bbox[0].sum()*rxy-side_length)
    xhat_max = 0.5*(bbox[0].sum()*rxy+side_length)

    yhat_min = 0.5*(bbox[1].sum()*rxy-side_length)
    yhat_max = 0.5*(bbox[1].sum()*rxy+side_length)

    zhat_min = 0.5*(bbox[2].sum()-side_length)
    zhat_max = 0.5*(bbox[2].sum()+side_length)

    assert np.allclose(np.r_[xhat_max-xhat_min,
                             yhat_max-yhat_min,
                             zhat_max-zhat_min], side_length)

    ########################################################
    # { Begin padding value and grids.
    pad = [[0,0], [0,0], [0,0]]
    for i in range(2):
        pad[i][0] = side_length / rxy
        pad[i][0] -= (bbox[i,1] - bbox[i,0])
        pad[i][0] = int(np.ceil(0.5*pad[i][0]))
        pad[i][1] = pad[i][0]

    # Compute x and y grids here. Z stuff further down.
    nx = bbox[0,1]-bbox[0,0]+1
    gridx = np.linspace(rxy*(bbox[0,0] - pad[0][0]),
                        rxy*(bbox[0,1] + pad[0][1]),
                        nx+2*pad[0][0])

    ny = bbox[1,1]-bbox[1,0]+1
    gridy = np.linspace(rxy*(bbox[1,0] - pad[1][0]),
                        rxy*(bbox[1,1] + pad[1][1]),
                        ny+2*pad[1][0])

    # *_start and *_stop are where the image will be placed
    # in the non-interpolated volume.
    xi_start = np.where(gridx >= 0)[0].min()
    xi_stop  = np.where(gridx <= rxy*511)[0].max()
    yi_start = np.where(gridy >= 0)[0].min()
    yi_stop  = np.where(gridy <= rxy*511)[0].max()

    # *low, and *high are the indices where the volume
    # is in the actual slices.
    xlow  = int(max(0,   bbox[0,0] - pad[0][0]))
    xhigh = int(min(511, bbox[0,1] + pad[0][1]))
    ylow  = int(max(0,   bbox[1,0] - pad[1][0]))
    yhigh = int(min(511, bbox[1,1] + pad[1][1]))

    # Determining the z-padding, etc, is not as easy...
    # A picture is really worth 1e3 words for this part.
    gridz = img_zs[np.logical_and(img_zs >= bbox[2,0],
                                  img_zs <= bbox[2,1])]
    if img_zs.min() <= zhat_min:
        betweenz  = np.logical_and(img_zs > zhat_min, \
                                   img_zs < bbox[2,0])
        pad[2][0] = betweenz.sum()+1
        gridz = np.r_[img_zs[betweenz], gridz]
        gridz = np.r_[img_zs[img_zs <= zhat_min].max(), gridz]
        zlow = np.abs(gridz[0]-img_zs).argmin()
        zi_offset = 0
    else:
        pad[2][0] = (img_zs < bbox[2,0]).sum()+2
        # In this case, the two previous grid points
        # don't matter as long as the first is <= zhat_min
        # Again, draw it out...
        z1 = zhat_min - ann.scan.slice_thickness
        z2 = 0.1*zhat_min + 0.9*img_zs.min()
        gridz = np.r_[z1, z2, img_zs[img_zs < bbox[2,0]], gridz]
        zlow = 0
        zi_offset = 2

    if img_zs.max() >= zhat_max:
        betweenz  = np.logical_and(img_zs > bbox[2,1], \
                                   img_zs < zhat_max)
        pad[2][1] = betweenz.sum()+1
        gridz = np.r_[gridz, img_zs[betweenz]]
        gridz = np.r_[gridz, img_zs[img_zs >= zhat_max].min()]
        zhigh = np.abs(gridz[-1]-img_zs).argmin()
    else:
        pad[2][1] = (img_zs > bbox[2,1]).sum()+2
        z1 = 0.9*img_zs.max() + 0.1*zhat_max
        z2 = zhat_max + ann.scan.slice_thickness
        gridz = np.r_[gridz, img_zs[img_zs > bbox[2,1]], z1, z2]
        zhigh = len(images)-1
    # } End padding value and grids.
    ########################################################

    cval = min([img.pixel_array.min() for img in images])

    # Initialize the nodule CT value volume.
    nodule = cval*np.ones(mask.shape)
    # Add the padding to both volumes.
    mask   = np.pad(mask, pad_width=pad, mode='constant',
                    constant_values=False)
    nodule = np.pad(nodule, pad_width=pad, mode='constant',
                    constant_values=cval)

    for k,z in enumerate(range(zlow,zhigh+1)):
        nodule[xi_start:xi_stop+1, yi_start:yi_stop+1, k+zi_offset] = \
            \
            images[z].pixel_array[xlow:xhigh+1,
                                  ylow:yhigh+1]

    # Now we create the interpolation grids.
    igridx,s = np.linspace(xhat_min, xhat_max, side_length+1, retstep=True)
    assert abs(s-1)<1e-6
    igridy,s = np.linspace(yhat_min, yhat_max, side_length+1, retstep=True)
    assert abs(s-1)<1e-6
    igridz,s = np.linspace(zhat_min, zhat_max, side_length+1, retstep=True)
    assert abs(s-1)<1e-6

    x,y,z = np.meshgrid(igridx, igridy, igridz, indexing='ij')
    X = np.c_[x.flatten(), y.flatten(), z.flatten()]
    s = igridx.shape[0]

    # Interpolate the nodule CT volume.
    rgi = RegularGridInterpolator(
            points=(gridx, gridy, gridz),
            values=nodule
          )
    inodule = rgi(X).reshape(s,s,s)

    # Interpolate the mask volume.
    rgi = RegularGridInterpolator(
            points=(gridx, gridy, gridz),
            values=mask
          )
    imask = rgi(X).reshape(s,s,s) > 0
    return inodule, imask

def get_img_zs(scan):
    imgs = scan.load_all_dicom_images()
    return np.array([float(im.ImagePositionPatient[-1]) for im in imgs])

