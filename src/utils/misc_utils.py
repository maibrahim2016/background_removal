import sys

if sys.version_info.major == 2:
    from base64 import decodestring as base64_decode
    from StringLikeIO import StringLikeIO as StringLikeIO
else:
    from base64 import decodebytes as base64_decode
    # Python 3 requires us to use BytesIO because the base64-decoding returns
    # a bytes object, so we mask it this way instead
    from io import BytesIO as StringLikeIO

import base64
import skimage.color
import skimage.io

def base64_to_image(base64_image):
    """Converts a base64-encoded image into a numpy array

    Parameters
    ----------
    base64_image : str
        Base64-encoded image.

    Returns
    -------
    image : instance of `numpy.ndarray`
        Image decoded from the provided base64 string.

        If the loaded image is greyscale and has only 2 channels, it will be
        converted to RGB format before being returned.

    """

    string_buffer = StringLikeIO(base64_decode(base64_image))

    image = skimage.io.imread(string_buffer)
    if image.ndim == 2:
        image = skimage.color.grey2rgb(image)

    return image

def image_to_base64(image):
    """Converts an image from a numpy array into its base64 encoding

    Parameters
    ----------
    image : instance of `numpy.ndarray`
        Image array to be encoded.

    Returns
    -------
    base64_image : str
        Base64-encoding of the image.

    """
    string_buffer = StringLikeIO()

    skimage.io.imsave(string_buffer, image)
    string_buffer.seek(0)

    base64_image = base64.b64encode(string_buffer.read()).decode('ascii')

    return base64_image
