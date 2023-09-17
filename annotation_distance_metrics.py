import numpy as np
from scipy.spatial.distance import cdist

metrics = {}

def pairdist(ann1, ann2, which):
    """
    Compute the pairwise euclidean distance between 
    the contour boundary points, and return the 
    minimum, maximum, or average value.

    which: str
        One of 'min', 'max', or 'avg'.
    """
    dists = cdist(ann1.contours_matrix,
                  ann2.contours_matrix)

    if   which == 'min':
        return dists.min()
    elif which == 'max':
        return dists.max()
    elif which == 'avg':
        return dists.mean()
    else:
        raise ValueError('invalid `which` value.')

metrics['min'] = lambda a,b: pairdist(a,b,'min')
metrics['max'] = lambda a,b: pairdist(a,b,'max')
metrics['avg'] = lambda a,b: pairdist(a,b,'avg')

def centroid_xyz(ann1, ann2):
    """
    Compute the euclidean distance between the x,y,z coordinates of
    each annotation's centroid.
    """
    return np.linalg.norm(ann1.centroid(0) - ann2.centroid(0))

metrics['centroid_xyz'] = centroid_xyz

def centroid_xy(ann1, ann2, which):
    """
    Get the distances between in-slice centroids for slices that
    have the same z value. If no z-value is shared between ann1 and
    ann2, then `centroid_xyz(ann1, ann2)` is returned.

    which: str
        One of 'min', 'max', or 'avg'.
    """
    P1 = ann1.contours_matrix
    P2 = ann2.contours_matrix

    zvals1 = set(np.unique(P1[:,2]).tolist())
    zvals2 = set(np.unique(P2[:,2]).tolist())
    zvals = zvals1.intersection(zvals2)

    if len(zvals) == 0:
        return centroid_xyz(ann1, ann2)

    dists = np.zeros(len(zvals))

    for i,z in enumerate(zvals):
        c1 = P1[:,:2][P1[:,2] == z].mean(0)
        c2 = P2[:,:2][P2[:,2] == z].mean(0) 
        dists[i] = np.linalg.norm(c1 - c2)

    if   which == 'min':
        return dists.min()
    elif which == 'max':
        return dists.max()
    elif which == 'avg':
        return dists.mean()
    else:
        raise ValueError('invalid `which` value.')

metrics['centroid_xy_min'] = lambda a,b: centroid_xy(a,b,'min')
metrics['centroid_xy_max'] = lambda a,b: centroid_xy(a,b,'max')
metrics['centroid_xy_avg'] = lambda a,b: centroid_xy(a,b,'avg')

def hausdorff(ann1, ann2):
    """
    Compute the Hausdorff distance [1] between the contour boundary points.

    [1]: https://en.wikipedia.org/wiki/Hausdorff_distance
    """
    C = cdist(ann1.contours_matrix,
              ann2.contours_matrix)
    return max(C.min(0).max(), C.min(1).max())

metrics['hausdorff'] = hausdorff

def jaccard(ann1, ann2):
    """
    The Jaccard distance [1] between the boolean volumes as point sets. The
    Jaccard distance is one minus the "intersection over union" score and is a
    value between 0 and 1. Distance 0 indicates perfect overlap
    (intersection/union = 1), while distance 1 indicates no overlap.

    [1]: https://en.wikipedia.org/wiki/Jaccard_index
    """
    A1 = ann1._as_set()
    A2 = ann2._as_set()
    I = len(A1.intersection(A2))
    return 1.0 - I*1.0 / len(A1.union(A2))

metrics['jaccard'] = jaccard
