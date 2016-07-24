# pylidc

This python module implements an object relational mapping (ORM) to an sqlite database containing the annotation information from the XML files provided by the LIDC dataset. This module is intended to make for easier data querying and to include functional aspects of the data models in addition to pure attribute information, e.g., computing nodule centroids from contour attributes.

:-:|:-:
![](https://raw.githubusercontent.com/pylidc/pylidc/master/img/viz-in-scan-example.png)|![](https://raw.githubusercontent.com/pylidc/pylidc/master/img/viz-in-3d-example.png)

See below for installation details and for examples.

## Installation and dependencies

### Installation

`pylidc` was developed in Linux. It should work in Mac OS X and Windows as well, but has not been tested yet. There may be minor peculiarities that need to be accounted for first. To install `pylidc` you should clone this repository:

    cd ~/Documents # or wherever you want to store the library
    git clone https://github.com/pylidc/pylidc.git

To let the python interpreter know about the module, add the path where you cloned the repository to your python path. For example, if you use bash, add

    export PYTHONPATH=$PYTHONPATH:~/Documents/pylidc

to your `bashrc`.

### Dependencies

Before we actually use the module, we have to make sure we have all the necessary dependencies.

The normal python scientific computing stack is required: `numpy`, `scipy`, and `matplotlib`. The ORM is implemented using `sqlalchemy`, so this is also required. Finally, the python dicom library is required. These can all be installed via pip:

    pip install numpy scipy matplotlib sqlalchemy pydicom

## Example usage

### Initial setup

The first thing you should do is tell the module where you store your dicom image files for LIDC dataset:

    >> import pylidc as pl
    >> pl.set_path_to_dicom_files('/path/to/big_external_drive/datasets/LIDC-IDRI')

The expected folder hierarchy in the specified path is the same as when you download the data from the TCIA download manager, i.e.,`PatientID` > `StudyInstanceUID` > `SeriesInstanceUID` > `*.dcm`. After you set this path initially, it will persist across sessions.

### Basic examples

There are three data models: `Scan`, `Annotation`, and `Contour`. The relationships are "one to many" for each model going left to right, i.e., `Scan`'s have many `Annotation`'s, and `Annotation`'s have many `Contour`'s. The main models to query are the `Scan` and `Annotation` models.

The main workhorse for querying is the `pylidc.query` function. This funciton just wraps the the `sqlalchemy.query` function. 

#### The `Scan` model

Here's some example usage for querying scan objects.

    import pylidc as pl
    
    qu = pl.query(pl.Scan).filter(pl.Scan.slice_thickness <= 1)
    print qu.count()
    # => 97

    scan = qu.first()
    print scan.patient_id, scan.pixel_spacing, scan.slice_thickness
    # => LIDC-IDRI-0066, 0.63671875, 0.6

    print len(scan.annotations)
    # => 11

    print scan.get_path_to_dicom_files()
    # '/path/to/big_external_drive/datasets/LIDC-IDRI/LIDC-IDRI-0066/1.3.6.1.4.1.14519.5.2.1.6279.6001.143774983852765282237869625332/1.3.6.1.4.1.14519.5.2.1.6279.6001.430109407146633213496148200410'

You can engage an interactive slice view by calling:

    scan.visualize()

Note that calling `visualize` on a scan object doesn't include its annotation information.

#### The `Annotation` model

Let's grab the first annotation from the `Scan` object above:

    ann = scan.annotations[0]
    print ann.scan.patient_id
    # => LIDC-IDRI-0066

    print ann.spiculation, ann.Spiculation()
    # => 3, Medium Spiculation

    print ann.estimate_diameter(), ann.estimate_volume()
    # => 15.4920358194, 888.052284241

    print ann.all_characteristics_as_string()
    # => Characteristic       Semantic value             # 
    # => -                    -                          - 
    # => Subtlety           | Low Subtlety             | 5 
    # => InternalStructure  | Soft Tissue              | 1 
    # => Calcification      | Absent                   | 6 
    # => Sphericity         | Ovoid                    | 3 
    # => Margin             | Poor                     | 1 
    # => Lobulation         | Medium-High Lobulation   | 4 
    # => Spiculation        | Medium Spiculation       | 3 
    # => Texture            | Solid Texture            | 5 
    # => Malignancy         | Medium-High Malignancy   | 4

Let's try a different query on the annotations directly:

    qu = pl.query(pl.Annotation).filter(pl.Annotation.lobulation > 3, pl.Annotation.malignancy == 5).f
    print qu.count()
    # => 183

    ann = qu.first()
    print ann.lobulation, ann.Lobulation(), ann.malignancy, ann.Malignancy()
    # => 4, Medium-High Lobulation, 5, High Malignancy

    print len(ann.contours)
    # => 46

    print ann.contours_to_matrix().shape
    # => (1754, 3)

    print ann.contours_to_matrix().mean(axis=0) - ann.centroid()
    # => [ 0.  0.  0.]

You can engage an interactive slice viewer that displays annotation values and the radiologist-drawn contours:

    ann.visualize_in_scan()

You can also view the nodule contours in 3d by calling:
    
    ann.visualize_in_3d()

Note that the 3d visualization works by assuming a roughly spherical shape, and may fail when this assumption is not true. This method is only meant for rough visualizations.

### More complicated examples / queries

#### Get a random result

One common objective for data exploration is to grab a random instance from some query. You can accomplish this by import `func` from `sqlalchemy`, and using `random`.

    from sqlalchemy import func
    scan = pl.query(pl.Scan).filter(pl.Scan.contrast_used == True).order_by(func.random()).first()
    ann  = pl.query(pl.Annotation).filter(pl.Annotation.malignancy == 5).order_by(func.random()).first()

The first query grabs a random `Scan` instance where contrast is used. The second query grabs a random `Annotation` instance where malignancy is equal to 5.

#### Query multiple model parameters with a join 

Another common objective is to query for an `Annotation` object which is constrained by its corresponding `Scan` in some way. For example:

    anns = pl.query(pl.Annotation).join(pl.Scan).filter(pl.Scan.slice_thickness < 1, pl.Annotation.malignancy != 3)
