import numpy as np
import pylidc as pl


scans = pl.query(pl.Scan)
nscans = scans.count()
for i,scan in enumerate(scans):
    print i+1,"/",nscans

    images = scan.load_all_dicom_images(verbose=0)
    img_zs = [float(img.ImagePositionPatient[-1]) for img in images]
    img_zs = np.unique(img_zs)

    for zval in img_zs:
        z = pl.Zval()
        z.val = float(zval)
        z.scan = scan

pl._session.commit()
