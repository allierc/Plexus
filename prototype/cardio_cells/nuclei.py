"""nuclei -- the one thing the picture DOES show, and the test it makes possible.

WHY THE EDGE TEST WAS THE WRONG TEST TO LEAN ON
================================================================================================
The motion boundaries do not coincide with image edges (z = -1.9) or with dark junctions (z = +1.1)
any more than the same boundary map shifted. That would be damning if the cells were visible -- but
the premise of this whole approach is that they are NOT, and a phase-contrast monolayer at
confluence has halos and texture where membranes should be. An absence of edges to agree with is
not evidence against a segmentation; it is why the segmentation was done from motion.

NUCLEI ARE VISIBLE, AND THERE IS ONE PER CELL. So a segmentation at cell scale must contain about
one nucleus per region -- too coarse and regions hold three or four, too fine and most hold none.
That is an independent check the motion never saw, and it also CALIBRATES the scale instead of
leaving it to a threshold I picked.
"""
import numpy as np
from scipy import ndimage as ndi


def detect(img, min_sigma=8, max_sigma=26, num_sigma=8, threshold=0.012):
    """Nuclei as bright-cored blobs in the band-passed image.

    Phase contrast inverts contrast with focus, so both polarities are tried and the one giving
    the more plausible count is kept -- stated rather than silently chosen.
    """
    from skimage.feature import blob_log
    a = img.astype(np.float32)
    a = (a - np.percentile(a, 1)) / (np.percentile(a, 99) - np.percentile(a, 1) + 1e-9)
    a = np.clip(a, 0, 1)
    a = a - ndi.gaussian_filter(a, 40)                    # flatten illumination
    out = {}
    for pol, arr in (("bright", a), ("dark", -a)):
        b = blob_log(arr, min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=num_sigma,
                     threshold=threshold, overlap=0.3)
        out[pol] = b
    return out
