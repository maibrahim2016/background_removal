import sys

if sys.version_info.major == 2:
    from base64 import decodestring as base64_decode
    from StringIO import StringLikeIO
else:
    from base64 import decodebytes as base64_decode
    # Python 3 requires us to use BytesIO because the base64-decoding returns
    # a bytes object, so we mask it this way instead
    from io import BytesIO as StringLikeIO

import base64
import numpy as np
import os
import six
import skimage.color
import skimage.io

from nose.tools import assert_is_instance
from numpy.testing import assert_array_almost_equal
from parameterized import parameterized
from PIL import Image

from .. import misc_utils

@parameterized.expand([
    [
        'colour_image',
        'test_colour_image.jpg',
    ],
    [
        'greyscale_image',
        'test_greyscale_image.jpg',
    ],
])
def test_base64_to_image(name, filename):
    filename = os.path.abspath(
        os.path.join(os.path.dirname(__file__), 'data', filename))
    
    with open(filename, 'rb') as fp:
        base64_image = base64.b64encode(fp.read())

    output = misc_utils.base64_to_image(base64_image)

    assert_is_instance(output, np.ndarray,
        msg='expected return value to be an instance of `numpy.ndarray`')

    image = skimage.color.grey2rgb(skimage.io.imread(filename))
    assert_array_almost_equal(image, output,
        err_msg='converted image is incorrect')

@parameterized.expand([
    [
        'colour_image',
        'test_colour_image.jpg',
    ],
    [
        'greyscale_image',
        'test_greyscale_image.jpg',
    ],
])
def test_image_to_base64(name, filename):
    filename = os.path.abspath(
        os.path.join(os.path.dirname(__file__), 'data', filename))
    
    image = skimage.color.grey2rgb(skimage.io.imread(filename))
    output = misc_utils.image_to_base64(image)

    assert_is_instance(output, six.string_types,
        msg='expected return value to be a string type')

    # seems like we don't really have a choice but to re-convert the base64-
    # encoded image into a numpy array for comparing; trying to use the raw
    # base64 strings don't seem to be able to match up properly
    stringlike_buffer = StringLikeIO(base64_decode(output.encode('utf-8')))
    loaded_image = Image.open(stringlike_buffer)
    assert_array_almost_equal(loaded_image, image,
        err_msg='converted image is incorrect')
