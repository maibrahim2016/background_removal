import sys

if sys.version_info.major == 2:
    import mock
else:
    from unittest import mock

import json
import numpy as np
import random
import string
import tensorflow as tf

from grpc._cython import cygrpc
from nose.tools import assert_equal
from nose.tools import assert_is_instance
from nose.tools import assert_raises
from nose.tools import assert_set_equal
from nose.tools import assert_true
from nose.tools import assert_tuple_equal
from numpy.testing import assert_array_almost_equal
from parameterized import parameterized
from tensorflow.contrib.util import make_tensor_proto
from tensorflow.python.framework import tensor_util

from .. import segmenter

def test_Segmenter_init_with_defaults():
    host, port = 'localhost', 8080
    model_name, signature_name = 'model', 'signature'
    input_name, output_name = 'input', 'output'

    model = segmenter.Segmenter(
        host,
        port,
        model_name,
        signature_name,
        input_name,
        output_name)

    expected_attributes_and_values = [
        ('host', host),
        ('port', port),
        ('model_name', model_name),
        ('signature_name', signature_name),
        ('input_name', input_name),
        ('output_name', output_name),
        ('request_timeout', 3.),
        ('max_send_message_length', None),
        ('max_receive_message_length', None),
    ]
    for (attribute, value) in expected_attributes_and_values:
        assert_true(hasattr(model, attribute),
            msg='expected model to have `{0}` attribute'.format(
                attribute))
        assert_equal(getattr(model, attribute), value,
            msg='incorrect value for `{0}` attribute'.format(
                attribute))

def test_Segmenter_init_with_provided_arguments():
    host, port = 'localhost', 8080
    model_name, signature_name = 'model', 'signature'
    input_name, output_name = 'input', 'output'

    request_timeout = random.randint(0, 1000)
    max_send_message_length = random.randint(0, 1000)
    max_receive_message_length = random.randint(0, 1000)

    model = segmenter.Segmenter(
        host,
        port,
        model_name,
        signature_name,
        input_name,
        output_name,
        request_timeout=request_timeout,
        max_send_message_length=max_send_message_length,
        max_receive_message_length=max_receive_message_length)

    expected_attributes_and_values = [
        ('host', host),
        ('port', port),
        ('model_name', model_name),
        ('signature_name', signature_name),
        ('input_name', input_name),
        ('output_name', output_name),
        ('request_timeout', request_timeout),
        ('max_send_message_length', max_send_message_length),
        ('max_receive_message_length', max_receive_message_length),
    ]
    for (attribute, value) in expected_attributes_and_values:
        assert_true(hasattr(model, attribute),
            msg='expected model to have `{0}` attribute'.format(
                attribute))
        assert_equal(getattr(model, attribute), value,
            msg='incorrect value for `{0}` attribute'.format(
                attribute))

@mock.patch(
    'src.models.segmenter.prediction_service_pb2_grpc.PredictionServiceStub',
    autospec=True)
@mock.patch(
    'src.models.segmenter.grpc.insecure_channel',
    autospec=True)
def test_Segmenter_channel_and_stub_creation(
    mock_insecure_channel, mock_PredictionServiceStub):

    host = ''.join(random.choice(string.ascii_letters) for _ in range(50))
    port = random.randint(0, int(10e6))

    model_name, signature_name = 'model', 'signature'
    input_name, output_name = 'input', 'output'

    _ = segmenter.Segmenter(
        host,
        port,
        model_name,
        signature_name,
        input_name,
        output_name)

    mock_insecure_channel.assert_called_once_with(
        target='{host}:{port}'.format(host=host, port=port),
        options=[])

    mock_PredictionServiceStub.assert_called_once_with(
        mock_insecure_channel.return_value)

@mock.patch(
    'src.models.segmenter.prediction_service_pb2_grpc.PredictionServiceStub',
    autospec=True)
@mock.patch(
    'src.models.segmenter.grpc.insecure_channel',
    autospec=True)
def test_Segmenter_channel_creation_with_options(
    mock_insecure_channel, mock_PredictionServiceStub):

    host, port = 'localhost', 8080
    model_name, signature_name = 'model', 'signature'
    input_name, output_name = 'input', 'output'

    max_send_message_length = random.randint(0, 1000)
    max_receive_message_length = random.randint(0, 1000)

    _ = segmenter.Segmenter(
        host,
        port,
        model_name,
        signature_name,
        input_name,
        output_name,
        max_send_message_length=max_send_message_length,
        max_receive_message_length=max_receive_message_length)

    expected_options = [
        (
            cygrpc.ChannelArgKey.max_send_message_length,
            max_send_message_length,
        ),
        (
            cygrpc.ChannelArgKey.max_receive_message_length,
            max_receive_message_length,
        ),
    ]
    mock_insecure_channel.assert_called_once_with(
        mock.ANY,
        options=expected_options)

@parameterized.expand([
    [
        'int32_dtype',
        np.random.randint(0, 256, (32, 200, 200, 3)).astype(np.int32),
        32, 200, 200, 1,
    ],
    [
        'uint8_dtype',
        np.random.randint(0, 256, (16, 512, 512, 3)).astype(np.uint8),
        16, 512, 512, 1,
    ],
    [
        'float64_dtype',
        np.random.rand(100, 227, 227, 3).astype(np.float64),
        100, 227, 227, 1,
    ],
])
def test_Segmenter_call_without_errors(
    name, images, expected_num_images, expected_height, expected_width,
    expected_channels):

    host, port = 'localhost', 8080
    model_name = ''.join(
        random.choice(string.ascii_letters) for _ in range(100))
    signature_name = ''.join(
        random.choice(string.ascii_letters) for _ in range(100))
    input_name = ''.join(
        random.choice(string.ascii_letters) for _ in range(100))
    output_name = ''.join(
        random.choice(string.ascii_letters) for _ in range(100))
    request_timeout = random.randint(100, 1000)

    model = segmenter.Segmenter(
        host,
        port,
        model_name,
        signature_name,
        input_name,
        output_name,
        request_timeout=request_timeout)

    mock_stub = mock.MagicMock(name='mock stub')
    model.stub = mock_stub

    mock_results = np.random.rand(
        expected_num_images,
        expected_height,
        expected_width,
        expected_channels)
    result = mock.MagicMock(
        name='mock future result',
        outputs={output_name: make_tensor_proto(mock_results.ravel())})

    mock_future = mock.MagicMock(name='mock future')
    mock_future.exception.return_value = None
    mock_future.result.return_value = result
    mock_stub.Predict.future.return_value = mock_future

    output = model(images)

    request, request_timeout = mock_stub.Predict.future.call_args[0]

    assert_equal(request.model_spec.name, model_name,
        msg='model name is incorrect')
    assert_equal(request.model_spec.signature_name, signature_name,
        msg='signature name is incorrect')

    input_names = [
        input_name,
    ]
    assert_set_equal(set(request.inputs.keys()), set(input_names),
        msg='expected input keys to be {0}, got {1}'.format(
            json.dumps(sorted(list(set(request.inputs.keys())))),
            json.dumps(sorted(list(set(input_names))))))

    expected_keys_and_values = [
        (
            input_name,
            images,
        ),
    ]
    for (key, value) in expected_keys_and_values:
        assert_array_almost_equal(
            tensor_util.MakeNdarray(request.inputs[key]),
            value,
            err_msg='incorrect value for "{0}"'.format(key))

    # special check for data type enforcement
    assert_equal(request.inputs[input_name].dtype, tf.float32,
        msg='expected the data type for "{0}" to be `tf.float32`')

    assert_is_instance(output, np.ndarray,
        msg='expected return value to be an instance of `numpy.ndarray`')

    assert_array_almost_equal(output, mock_results,
        err_msg='return value is incorrect')

def test_Segmenter_call_with_exception():
    host, port = 'localhost', 8080
    model_name, signature_name = 'model', 'signature'
    input_name, output_name = 'input', 'output'

    model = segmenter.Segmenter(
        host,
        port,
        model_name,
        signature_name,
        input_name,
        output_name)

    mock_stub = mock.MagicMock(name='mock stub')
    model.stub = mock_stub

    mock_future = mock.MagicMock(name='mock future')
    mock_error = RuntimeError('mock future error')
    mock_future.exception.return_value = mock_error
    mock_stub.Predict.future.return_value = mock_future

    images = np.random.rand(16, 200, 200, 3)
    assert_raises(type(mock_error), model, images)

@parameterized.expand([
    [
        'within_bounds',
        1000,
        (400, 700, 3),
        (400, 700, 3),
    ],
    [
        'width_out_of_bounds',
        400,
        (300, 500, 3),
        (240, 400, 3),
    ],
    [
        'height_out_of_bounds',
        400,
        (800, 300, 3),
        (400, 150, 3),
    ],
    [
        'both_width_and_height_out_of_bounds',
        500,
        ( 600,  800, 3),
        ( 375,  500, 3),
    ],
])
@mock.patch(
    'src.models.segmenter.Segmenter.__init__',
    autospec=False,
    return_value=None)
def test_Segmenter_aspect_aware_resizing(
    name, max_size, input_dims, expected_dims, mock_Segmenter):

    model = segmenter.Segmenter()

    images = np.random.randint(0, 256, input_dims).astype(np.uint8)

    output = model.aspect_aware_resizing(images, max_size)

    assert_is_instance(output, np.ndarray,
        msg='expected return value to be an instance of `numpy.ndarray`')

    assert_tuple_equal(output.shape, expected_dims,
        msg='resized image dimensions are incorrect')

@mock.patch(
    'src.models.segmenter.Image.fromarray',
    autospec=True)
@mock.patch(
    'src.models.segmenter.Segmenter.__init__',
    autospec=False,
    return_value=None)
def test_Segmenter_aspect_aware_resizing_interpolation(
    mock_Segmenter, mock_fromarray):

    max_size = 100
    interpolation = mock.MagicMock(name='mock interpolation enum')
    mock_image = mock.MagicMock(name='mock image', shape=(256, 256))

    mock_pillow_image = mock.MagicMock(name='mock Pillow image')
    mock_resized_image = np.random.rand(max_size, max_size)
    mock_pillow_image.resize.return_value = mock_resized_image
    mock_fromarray.return_value = mock_pillow_image

    model = segmenter.Segmenter()

    output = model.aspect_aware_resizing(
        mock_image,
        max_size,
        interpolation=interpolation)

    mock_pillow_image.resize.assert_called_once_with(
        (max_size, max_size),
        resample=interpolation)
