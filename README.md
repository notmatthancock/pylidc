# pylidc

This python module implements an object relational mapping (ORM) to an sqlite database containing the annotation information from the XML files provided by the LIDC dataset. This module is intended to make for easier data querying and to include functional aspects of the data models in addition to pure attribute information, e.g., computing nodule centroids from contour attributes.

### Visualize contour annotations on top of CT scan data:
[[http://github.com/pylidc/pylidc/blob/master/img/viz-in-scan-example.png]]

### Visualize contour annotations in 3d:
[[http://github.com/pylidc/pylidc/blob/master/img/viz-in-scan-3d.png]]

See below for installation details and for examples.

## Installation and dependencies



## Examples

The ORM is implemented using sqlalchemy. There are three data models: `Scan`, `Annotation`, and `Contour`. The relationships are "one to many" for each model going left to right, i.e., `Scan`'s have many `Annotation`'s, and `Annotation`'s have many `Contour`'s.

