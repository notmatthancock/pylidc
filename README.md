
# pylidc

`pylidc` is a python library intended to enhance workflow associated with the [LIDC dataset](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI), including utilities for both querying by attributes (e.g., collecting all annotations where malignancy is labeled as greater than 3 and spiculation is labeled a value equal to 1), and for common functional routines that act on the associated data (e.g., estimating the diameter or volume of a nodule from one of its annotations). 

Routines for visualizing the annotations, both atop the CT data and as a surface in 3D, are implemented. These functionalities are implemented via an object relational mapping (ORM), using `sqlalchemy` to an sqlite database containing the annotation information from the XML files provided by the LIDC dataset.

![](https://raw.githubusercontent.com/pylidc/pylidc/master/img/viz-in-scan-example.png)
![](https://raw.githubusercontent.com/pylidc/pylidc/master/img/viz-in-3d-example.png)

Table of Contents
=================

  * [Installation and setup](#installation-and-setup)
     * [Dicom file directory configuration](#dicom-file-directory-configuration)
  * [Citing](#citing)
  * [Example usage](#example-usage)
     * [Basic examples](#basic-examples)
        * [The Scan model](#the-scan-model)
        * [The Annotation model](#the-annotation-model)
     * [Advanced queries](#advanced-queries)
        * [Get a random result](#get-a-random-result)
        * [Query multiple model parameters with a join](#query-multiple-model-parameters-with-a-join)
     * [Resampling the volumes](#resampling-the-volumes)
     * [Clustering annotations](#clustering-annotations-to-identify-those-which-refer-to-the-same-physical-nodule)


## Installation and setup

`pylidc` has been used on Linux, Mac, and Windows, and on Python 2 and 3.

The package can be installed via `pip`:

    pip install pylidc

### Dicom file directory configuration

While `pylidc` has many functions for analyzing and querying only annotation data (which do not require DICOM image data access), `pylidc` also has many functions that do require access to the DICOM files associated with the LIDC dataset. `pylidc` looks for a special configuration file that tells it where DICOM data is located on your system. You can use `pylidc` without creating this configuration file, but of course, any functions that depend on CT image data will not be usable.

`pylidc` looks in your home folder for a configuration file called, `.pylidcrc` on Mac and Linux, or `pylidc.conf` on Windows. You must create this file. On Linux and Mac, the file should be located at `/home/[user]/.pylidcrc`. On Windows, the file should be located at `C:\Users\[User]\pylidc.conf`.

The configuration file should be formatted as follows:

    [dicom]
    path = /path/to/big_external_drive/datasets/LIDC-IDRI
    warn = True

If you want to use `pylidc` without utilizing the DICOM data (for say, querying annotation attributes, etc.), you can remove `path` and set `warn` to `False`,  i.e.,

    [dicom]
    warn = False

and the module won't bother you about it each time you import the module.

The expected folder hierarchy in the specified `path` is: `PatientID` > `StudyInstanceUID` > `SeriesInstanceUID` > `*.dcm`. If you downloaded the data from the TCIA site, the folder hierarchy will (probably!) already be formatted in this way.

## Citing

If you find `pylidc` helpful to your research, you could cite by noting that software was developed for, and first mentioned in, the following publication:

> Matthew C. Hancock, Jerry F. Magnan. **Lung nodule malignancy classification using only radiologist quantified image features as inputs to statistical learning algorithms: probing the Lung Image Database Consortium dataset with two statistical learning methods**. *SPIE Journal of Medical Imaging*. Dec. 2016. [http://dx.doi.org/10.1117/1.JMI.3.4.044504](http://dx.doi.org/10.1117/1.JMI.3.4.044504)

## Example usage

### Basic examples

There are three data models: `Scan`, `Annotation`, and `Contour`. The relationships are "one to many" for each model going left to right, i.e., `Scan`'s have many `Annotation`'s, and `Annotation`'s have many `Contour`'s. The main models to query are the `Scan` and `Annotation` models.

The main workhorse for querying is the `pylidc.query` function. This function just wraps the `sqlalchemy.query` function. 

#### The `Scan` model

Here's some example usage for querying scan objects.

    import pylidc as pl
    
    qu = pl.query(pl.Scan).filter(pl.Scan.slice_thickness <= 1)
    print(qu.count())
    # => 97

    scan = qu.first()
    print(scan.patient_id, scan.pixel_spacing, scan.slice_thickness)
    # => LIDC-IDRI-0066, 0.63671875, 0.6

    print(len(scan.annotations))
    # => 11

    print(scan.get_path_to_dicom_files())
    # '/path/to/big_external_drive/datasets/LIDC-IDRI/LIDC-IDRI-0066/1.3.6.1.4.1.14519.5.2.1.6279.6001.143774983852765282237869625332/1.3.6.1.4.1.14519.5.2.1.6279.6001.430109407146633213496148200410'

You can engage an interactive slice view by calling:

    scan.visualize()

By default, note that calling `visualize` on a scan object doesn't include its annotation information. However, if `scan.visualize` is supplied with a list of `Annotation` objects (say, grouped by the `scan.cluster_annotations()` function), then this annotation information *will* be displayed. For example

    nodules = scan.cluster_annotations()
    print(len(nodules))
    # => 3

    # Visualize the slices with the provided annotations indicated by arrows.
    scan.visualize(annotations)

`scan.cluster_annotations` returns a list. Each element of the list is a list of `Annotation` objects where each `Annotation` refers to the same nodule in the scan (probably). See [Clustering annotations](#clustering-annotations-to-identify-those-which-refer-to-the-same-physical-nodule) for more details on this function.


#### The `Annotation` model

Let's grab the first annotation from the `Scan` object above:

    ann = scan.annotations[0]
    print(ann.scan.patient_id)
    # => LIDC-IDRI-0066

    print(ann.spiculation, ann.Spiculation)
    # => 3, Medium Spiculation

    print("%.2f, %.2f, %.2f" % (ann.diameter, ann.surface_area, ann.volume))
    # => 15.49, 1041.37, 888.05

    ann.print_formatted_feature_table()
    # => Feature              Meaning                    # 
    # => -                    -                          - 
    # => Subtlety           | Obvious                  | 5 
    # => Internalstructure  | Soft Tissue              | 1 
    # => Calcification      | Absent                   | 6 
    # => Sphericity         | Ovoid                    | 3 
    # => Margin             | Poorly Defined           | 1 
    # => Lobulation         | Near Marked Lobulation   | 4 
    # => Spiculation        | Medium Spiculation       | 3 
    # => Texture            | Solid                    | 5 
    # => Malignancy         | Moderately Suspicious    | 4

    fvals, fstrings = ann.feature_vals(return_str=True)

    for fname,fval,fstr in zip(pl.annotation_feature_names, fvals, fstrings):
        print(fname.title(), fval, fstr)
    # => 'Subtlety', 5, 'Obvious'
    # => 'Internalstructure', 1, 'Soft Tissue'
    # => 'Calcification', 6, 'Absent'
    # => 'Sphericity', 3, 'Ovoid'
    # => 'Margin', 1, 'Poorly Defined'
    # => 'Lobulation', 4, 'Near Marked Lobulation'
    # => 'Spiculation', 3, 'Medium Spiculation'
    # => 'Texture', 5, 'Solid'
    # => 'Malignancy', 4, 'Moderately Suspicious'

Let's try a different query on the annotations directly:

    qu = pl.query(pl.Annotation).filter(pl.Annotation.lobulation > 3,
                                        pl.Annotation.malignancy == 5)
    print(qu.count())
    # => 183

    ann = qu.first()
    print(ann.lobulation, ann.Lobulation, ann.malignancy, ann.Malignancy)
    # => 4, Near Marked Lobulation, 5, Highly Suspicious

    print(len(ann.contours))
    # => 8

    print(ann.contours_to_matrix().shape)
    # => (671, 3)

    print(ann.contours_to_matrix().mean(axis=0) - ann.centroid)
    # => [ 0.  0.  0.]

You can engage an interactive slice viewer that displays annotation values and the radiologist-drawn contours:

    ann.visualize_in_scan()

You can also view the nodule contours in 3d by calling:
    
    ann.visualize_in_3d()

### Advanced queries

#### Get a random result

One common objective for data exploration is to grab a random instance from some query. You can accomplish this by import `func` from `sqlalchemy`, and using `random`.

    from sqlalchemy import func
    scan = pl.query(pl.Scan).filter(pl.Scan.contrast_used == True).order_by(func.random()).first()
    ann  = pl.query(pl.Annotation).filter(pl.Annotation.malignancy == 5).order_by(func.random()).first()

The first query grabs a random `Scan` instance where contrast is used. The second query grabs a random `Annotation` instance where malignancy is equal to 5.

#### Query multiple model parameters with a join 

Another common objective is to query for an `Annotation` object which is constrained by its corresponding `Scan` in some way. For example:

    anns = pl.query(pl.Annotation).join(pl.Scan).filter(pl.Scan.slice_thickness < 1, pl.Annotation.malignancy != 3)

### Resampling the volumes

The `Annotation` member function, `uniform_cubic_resample`, takes a cubic region of interest with the centroid at the center of the volume. The corresponding CT value volume is resampled to have voxel spacing of 1 millimeter and a side length as given by the functions `side_length` parameter. Along with the uniformly resampled, cubic CT image volume, a corresponding boolean-valued volume is also returned that is 1 where the nodule exists in the resampled CT volume and 0 otherwise.
    
    import pylidc as pl
    import matplotlib.pyplot as plt
    from skimage.measure import find_contours

    ann = pl.query(pl.Annotation).first()
    vol, seg = ann.uniform_cubic_resample(side_length = 100)
    print(vol.shape, seg.shape)
    # => (101, 101, 101) (101, 101, 101)

    # View middle slice of interpolated volume (pixel spacing now = 1mm)
    plt.imshow(vol[:,:,50], cmap=plt.cm.gray)

    # View middle slice of interpolated segmentation volume as contours
    # atop the interpolated image.
    contours = find_contours(seg[:,:,50], 0.5)
    for contour in contours:
        plt.plot(contour[:,1], contour[:,0], '-r')

    plt.show()

![](https://raw.githubusercontent.com/pylidc/pylidc/master/img/resample-example.png)

### Clustering annotations to identify those which refer to the same physical nodule.

The LIDC dataset doesn't assign unique global identifiers to the physical nodules. For a given physical nodule, there may exist up to 4 annotations that refer to it. The annotations are anonymous, so even if it is known that 4 annotations refer to the same nodule, it is impossible to tell which annotator provided each annotation across multiple nodules consistently.

However, we can estimate when annotations refer to the same physical nodule in a scan by examining the properties of the annotations and clustering them based on the properties. `pylidc` provides [a number of distance metrics between annotations](https://github.com/pylidc/pylidc/pylidc/annotation_distance_metrics.py) based on the annotation contour coordinates. The `Scan` model provides a [`cluster_annotations`](https://github.com/pylidc/pylidc/pylidc/Scan.py) function which then clusters annotations by determining the connected components of the adjacency graph associated with a chosen distance-between-annotations metric and a chosen distance tolerance.

Here's an example:

```
import pylidc as pl

scan = pl.query(pl.Scan).first()
nods = scan.cluster_annotations()

print("Scan is estimated to have", len(nods), "nodules.")

for i,nod in enumerate(nods):
    print("Nodule", i+1, "has", len(nod), "annotations.")
        for j,ann in enumerate(nod):
            print("-- Annotation", j+1, "centroid:", ann.centroid)
```

Output:
```
Scan is estimated to have 4 nodules.
Nodule 1 has 4 annotations.
-- Annotation 1 centroid: [  331.90680101   312.30982368  1480.44962217]
-- Annotation 2 centroid: [  328.60546875   309.91796875  1479.73046875]
-- Annotation 3 centroid: [  327.91666667   309.88293651  1479.01785714]
-- Annotation 4 centroid: [  332.55660377   313.88050314  1479.94339623]
Nodule 2 has 4 annotations.
-- Annotation 1 centroid: [  360.81122449   169.19642857  1542.10459184]
-- Annotation 2 centroid: [  360.82233503   169.21319797  1542.14720812]
-- Annotation 3 centroid: [  361.05243446   168.86142322  1542.34269663]
-- Annotation 4 centroid: [  361.25501433   171.          1542.80659026]
Nodule 3 has 1 annotations.
-- Annotation 1 centroid: [  336.41666667   348.83333333  1545.75      ]
Nodule 4 has 4 annotations.
-- Annotation 1 centroid: [  340.54020979   245.07692308  1606.14160839]
-- Annotation 2 centroid: [  341.29061103   244.65275708  1605.90834575]
-- Annotation 3 centroid: [  341.75417299   244.03490137  1606.95827011]
-- Annotation 4 centroid: [  341.53110048   245.58532695  1606.5       ]
```
You can supply annotation clusters (variable, `nods`, above) to the `scan.visualize` function, and arrows will annotate where the nodules are present in the scan.
