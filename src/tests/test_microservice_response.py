import sys

if sys.version_info.major == 2:
    import httplib
    from base64 import decodestring as base64_decode
    from StringIO import StringIO as StringLikeIO
else:
    from http import client as httplib
    from base64 import decodebytes as base64_decode
    from io import BytesIO as StringLikeIO

import base64
import json
import numpy as np
import os
import random
import requests
import six
import skimage.io
import skimage.transform
import warnings

from collections import Mapping
from nose.tools import assert_equal
from nose.tools import assert_in
from nose.tools import assert_is_instance
from nose.tools import assert_true
from nose.tools import assert_tuple_equal
from numpy.testing import assert_array_almost_equal
from parameterized import parameterized
from unittest import TestSuite

def bounded_repr(obj, limit=100, trim=20):
    obj_repr = repr(obj)

    if len(obj_repr) > limit:
        obj_repr = '{0}...<truncated>...{1}'.format(
            obj_repr[:trim],
            obj_repr[-trim:])
    
    return obj_repr

def assert_response_expectations(
    response, expected_status_codes, ok_key, expected_ok_value):

    expected_status_codes = set(expected_status_codes)

    assert_in(response.status_code, expected_status_codes,
        msg='got invalid status code HTTP {0} from endpoint'.format(
            response.status_code))

    json_data = response.json()

    assert_in(ok_key, json_data,
        msg='expected "{0}" key to be present in response JSON data'.format(
            ok_key))
    
    assert_equal(json_data[ok_key], expected_ok_value,
        msg='expected "{0}" key to be {1}, got {2} (type={3})'.format(
            ok_key,
            expected_ok_value,
            repr(json_data[ok_key]),
            type(json_data[ok_key]).__name__))

    return json_data

def assert_follows_schema(schema, data, name):

    if isinstance(schema, type):
        assert_is_instance(data, schema,
            msg=('expected `{name}` to be of type {expected_type}, got '
                 '{actual_type} [{repr}]').format(
                    name=name,
                    expected_type=schema,
                    actual_type=type(data).__name__,
                    repr=bounded_repr(data)))

    elif isinstance(schema, tuple):
        assert_true(isinstance(data, schema),
            msg=('expected `{name}` to be one of types {expected_types}, got '
                 '{actual_type} [{repr}]').format(
                    name=name,
                    expected_types=json.dumps([t.__name__ for t in schema]),
                    actual_type=type(data).__name__,
                    repr=bounded_repr(data)))

    elif isinstance(schema, Mapping):
        for schema_key, schema_value in six.iteritems(schema):
            assert_follows_schema(
                schema_value,
                data[schema_key], '{name}.{key}'.format(
                    name=name, key=schema_key))

    elif isinstance(schema, list):
        
        if len(schema) != 1:
            raise ValueError(
                ('schema assertions for lists are expected to contain only 1 '
                 'element, but {num_elements} elements were provided').format(
                    num_elements=len(schema)))

        for i, element in enumerate(data):
            assert_follows_schema(
                schema[0],
                element,
                '{name}.#{num}'.format(name=name, num=i))

    else:
        raise ValueError(
            ('got unsupported schema definition, got type {schema_type} '
             'instead').format(schema_type=type(schema).__name__))


class TestSegmenter(TestSuite):

    def setUp(self):
        host = os.environ.get('SEGMENTER_TEST_HOST', 'http://localhost')
        port = os.environ.get('SEGMENTER_TEST_PORT', 80)
        self.max_size = os.environ.get('SEGMENTER_TEST_MAX_SIZE', 1080)

        endpoint = '{host}:{port}'.format(host=host, port=port)
        response = requests.get(endpoint, timeout=3)

        if response.status_code != httplib.OK:
            raise RuntimeError(
                'got HTTP {0} from microservice at {1}'.format(
                    response.status_code,
                    endpoint))

        before_filename = os.path.abspath(
            os.path.join(os.path.dirname(__file__), 'data', 'before.png'))
        if not os.path.isfile(before_filename):
            raise RuntimeError(
                'could not locate "before" test image [{0}]'.format(
                    before_filename))

        after_filename = os.path.abspath(
            os.path.join(os.path.dirname(__file__), 'data', 'after.npy'))
        if not os.path.isfile(after_filename):
            raise RuntimeError(
                'could not locate "after" test image [{0}]'.format(
                    after_filename))

        self.endpoint = endpoint
        self.before_filename = before_filename
        self.after_filename = after_filename

    def test_ping_response(self):
        response = requests.get(self.endpoint)

        assert_equal(response.status_code, httplib.OK,
            msg='expected HTTP {0}, got HTTP {1}'.format(
                httplib.OK, response.status_code))

    def test_inference_with_single_image(self):
        before_image = skimage.io.imread(self.before_filename)
        with open(self.before_filename, 'rb') as fp:
            base64_before_image = base64.b64encode(fp.read()).decode('ascii')

        response = requests.post(
            self.endpoint,
            data={
                'images': json.dumps([base64_before_image]),
            })

        json_data = assert_response_expectations(
            response,
            expected_status_codes=[httplib.OK],
            ok_key='ok',
            expected_ok_value=True)

        expected_schema = {
            'ok': bool,
            'masks': six.string_types,
        }
        assert_follows_schema(expected_schema, json_data, 'response')

        returned_mask_string = json.loads(json_data['masks'])
        stringlike_buffers = [
            StringLikeIO(base64_decode(base64_response.encode('ascii')))
                for base64_response in returned_mask_string]
        returned_masks = [skimage.io.imread(b) for b in stringlike_buffers]

        assert_equal(len(returned_masks), 1,
            msg='expected only 1 mask in the response')

        returned_mask = returned_masks[0]

        after_mask = np.load(self.after_filename)
        assert_array_almost_equal(after_mask, returned_mask,
            err_msg='segmentation mask is incorrect')

    def test_inference_with_multiple_images(self):
        num_images = random.randint(2, 5)

        before_image = skimage.io.imread(self.before_filename)
        with open(self.before_filename, 'rb') as fp:
            base64_before_image = base64.b64encode(fp.read()).decode('ascii')

        response = requests.post(
            self.endpoint,
            data={
                'images': json.dumps(
                    [base64_before_image for _ in range(num_images)]),
            })

        json_data = assert_response_expectations(
            response,
            expected_status_codes=[httplib.OK],
            ok_key='ok',
            expected_ok_value=True)

        expected_schema = {
            'ok': bool,
            'masks': six.string_types,
        }
        assert_follows_schema(expected_schema, json_data, 'response')

        returned_mask_string = json.loads(json_data['masks'])
        stringlike_buffers = [
            StringLikeIO(base64_decode(base64_response.encode('ascii')))
                for base64_response in returned_mask_string]
        returned_masks = [skimage.io.imread(b) for b in stringlike_buffers]

        assert_equal(len(returned_masks), num_images,
            msg='expected {0} masks in the response, got {1}'.format(
                num_images, len(returned_masks)))

        after_mask = np.load(self.after_filename)

        for returned_mask in returned_masks:

            assert_array_almost_equal(
                after_mask,
                returned_mask,
                err_msg='segmentation mask is incorrect')

    def test_inference_with_oversized_image(self):
        before_image = skimage.io.imread(self.before_filename)

        # catch skimage's warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=UserWarning)

            before_image = skimage.transform.resize(
                before_image,
                (int(1.1 * self.max_size), int(1.1 * self.max_size)))

            stringlike_buffer = StringLikeIO()
            skimage.io.imsave(stringlike_buffer, before_image)

        base64_before_image = base64.b64encode(
            stringlike_buffer.getvalue()).decode('ascii')

        response = requests.post(
            self.endpoint,
            data={
                'images': json.dumps([base64_before_image]),
            })

        json_data = assert_response_expectations(
            response,
            expected_status_codes=[httplib.OK],
            ok_key='ok',
            expected_ok_value=True)

        expected_schema = {
            'ok': bool,
            'masks': six.string_types,
        }
        assert_follows_schema(expected_schema, json_data, 'response')

        returned_mask_string = json.loads(json_data['masks'])
        stringlike_buffers = [
            StringLikeIO(base64_decode(base64_response.encode('ascii')))
                for base64_response in returned_mask_string]
        returned_masks = [skimage.io.imread(b) for b in stringlike_buffers]

        assert_equal(len(returned_masks), 1,
            msg='expected only 1 mask in the response')

        returned_mask = returned_masks[0]

        assert_tuple_equal(returned_mask.shape, before_image.shape[:2],
            msg='expected mask shape to be {0}, got {1}'.format(
                before_image.shape[:2], returned_mask.shape))

    @parameterized.expand([
        [
            'mising_images',
            {},
        ],
        [
            'no_images',
            {
                'images': json.dumps([]),
            },
        ],
    ])
    def test_inference_with_missing_parameters(self, name, payload):
        response = requests.post(self.endpoint, data=payload)

        json_data = assert_response_expectations(
            response,
            expected_status_codes=[httplib.BAD_REQUEST],
            ok_key='ok',
            expected_ok_value=False)

        expected_schema = {
            'ok': bool,
            'error_message': six.string_types,
        }
        assert_follows_schema(expected_schema, json_data, 'response')
