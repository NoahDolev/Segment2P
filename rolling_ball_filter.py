import numpy as np
import scipy.ndimage as ndi
from scipy.ndimage._ni_support import _normalize_sequence

def rolling_ball_filter(data, ball_radius, spacing=None, top=False, **kwargs):
    """Rolling ball filter implemented with morphology operations

    This implenetation is very similar to that in ImageJ and uses a top hat transform
    with a ball shaped structuring element
    https://en.wikipedia.org/wiki/Top-hat_transform

    Parameters
    ----------
    data : ndarray
        image data (assumed to be on a regular grid)
    ball_radius : float
        the radius of the ball to roll
    spacing : int or sequence
        the spacing of the image data
    top : bool
        whether to roll the ball on the top or bottom of the data
    kwargs : key word arguments
        these are passed to the ndimage morphological operations

    Returns
    -------
    data_nb : ndarray
        data with background subtracted
    bg : ndarray
        background that was subtracted from the data
    """
    ndim = data.ndim
    if spacing is None:
        spacing = np.ones_like(ndim)
    else:
        spacing = _normalize_sequence(spacing, ndim)

    radius = np.asarray(_normalize_sequence(ball_radius, ndim))
    mesh = np.array(np.meshgrid(*[np.arange(-r, r + s, s) for r, s in zip(radius, spacing)], indexing="ij"))
    structure = 2 * np.sqrt(1 - ((mesh / radius.reshape(-1, *((1,) * ndim)))**2).sum(0))
    structure[~np.isfinite(structure)] = 0
    if not top:
        # ndi.white_tophat(y, structure=structure, output=background)
        background = ndi.grey_erosion(data, structure=structure, **kwargs)
        background = ndi.grey_dilation(background, structure=structure, **kwargs)
    else:
        # ndi.black_tophat(y, structure=structure, output=background)
        background = ndi.grey_dilation(data, structure=structure, **kwargs)
        background = ndi.grey_erosion(background, structure=structure, **kwargs)

    return data - background, background