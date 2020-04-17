import grpc
import numpy as np
import tensorflow as tf

from grpc._cython import cygrpc
from PIL import Image
from tensorflow.contrib.util import make_tensor_proto
from tensorflow.python.framework import tensor_util
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc

from crf import dense_crf
from postprocessing import postprocess
import cv2

class Segmenter(object):
    """Proxy model for the segmentation algorithm

    This class is an adapter meant to be used in front of a Tensorflow Serving
    model server, using the RPC protocol.

    Parameters
    ----------
    host : str
        Model server host name.
    port : int
        Model server port.
    model_name : str
        Name of model assigned, which should correspond to the --model_name
        flag used at model server start up.
    signature_name : str
        Signature definition used at model saving time. This needs to come from
        the individual who created the model.
    input_name : str
        Key assigned to the required input image placeholder, also defined at
        model saving time.
    output_name : str
        Key assigned to the required output image operation, also defined at
        model saving time.
    request_timeout : int or float, optional, defaults to 3
        Request timeout when forwarding the request to the model server.
    max_send_message_length : int, optional, defaults to None
        Refers to the maximum size limit of the package transmitted to the
        model server.

        If None is specified, will not specify the option at channel creation
        time. Otherwise, the specified value will be set. Specify -1 for
        unlimited, which is recommended in the case of images.
    max_receive_message_length : int, optional, defaults to None
        Refers to the maximum size limit of the package transmitted from the
        model server.

        If None is specified, will not specify the option at channel creation
        time. Otherwise, the specified value will be set. Specify -1 for
        unlimited, which is recommended in the case of images.

    Attributes
    ----------
    host : str
        Refer to `host` parameter.
    port : int
        Refer to `port` parameter.
    model_name : str
        Refer to `model_name` parameter.
    signature_name : str
        Refer to `signature_name` parameter.
    input_name : str
        Refer to `input_name` parameter.
    output_name : str
        Refer to `output_name` parameter.
    request_timeout : int or float
        Refer to `request_timeout` parameter.
    max_send_message_length : int or None
        Refer to `max_send_message_length` parameter.
    max_receive_message_length : int or None
        Refer to `max_receive_message_length` parameter.
    channel : instance of `grpc._channel.Channel`
        Channel object from grpc used to create the stub for communicating with
        the model server.
    stub : instance of `prediction_service_pb2_grpc.PredictionServiceStub`
        Stub object supplied by the Tensorflow Serving SDK, used to create
        requests to the model server.

    Notes
    -----
    The specifications for this model was built based on the example provided
    by Brian, the owner of the research project.

    In the event that the number of input or output placeholders changes, there
    will have to be modifications.

    """
    
    def __init__(
        self, host, port, model_name, signature_name, input_name, output_name, 
        threshold, request_timeout=3., max_send_message_length=None,
        max_receive_message_length=None):

        self.host = host
        self.port = port

        self.model_name = model_name
        self.signature_name = signature_name

        self.input_name = input_name
        self.output_name = output_name

        self.max_send_message_length = max_send_message_length
        self.max_receive_message_length = max_receive_message_length

        self.threshold = threshold

        channel_options = []

        if self.max_send_message_length is not None:
            channel_options.append((
                cygrpc.ChannelArgKey.max_send_message_length,
                self.max_send_message_length,
            ))

        if self.max_receive_message_length is not None:
            channel_options.append((
                cygrpc.ChannelArgKey.max_receive_message_length,
                self.max_receive_message_length,
            ))

        self.channel = grpc.insecure_channel(
            target='{host}:{port}'.format(host=self.host, port=self.port),
            options=channel_options)

        self.stub = prediction_service_pb2_grpc.PredictionServiceStub(
            self.channel)
        self.request_timeout = request_timeout

    def __call__(self, image, images_tensor, height, width):
        """Forwards a request to the model server

        Parameters
        ----------
        images : instance of `numpy.ndarray`
            4-dimensional RGB image matrix of shape
            (num images, height, width, channels) normalised to the [0, 1]
            interval.

            Brian has decided not to support batch processing, but given that
            the model was trained with the ability to accept batches of images,
            we still allow for it within the proxy model itself, but not the
            microservice. If you wish to implement actual batch processing, you
            have to first determine your preferred method of ensure that the
            entire batch of images are of the same dimensions first.

        Returns
        -------
        output_masks : instance of `numpy.ndarray`
            Model inference results, with dimensions
            (num images, height, width, 1).

        """

        shape = images_tensor.shape
        batch = images_tensor.tolist()

        request = predict_pb2.PredictRequest()

        request.model_spec.name = self.model_name
        request.model_spec.signature_name = self.signature_name

        # XXX For now, since all the models in the ensembled model use the same backbone/encoder, "efficientnetb5", 
        # I'm just duplicating the input tensor to the three different outputs.
        input_node_name_0 = "input_fpn_efficientnetb5"
        input_node_name_1 = "input_unet_efficientnetb5"
        input_node_name_2 = "input_linknet_efficientnetb5"
        request.inputs[input_node_name_0].CopyFrom(make_tensor_proto(batch, shape=shape))
        request.inputs[input_node_name_1].CopyFrom(make_tensor_proto(batch, shape=shape))
        request.inputs[input_node_name_2].CopyFrom(make_tensor_proto(batch, shape=shape))

        result = self.stub.Predict(request, self.request_timeout)
        # result, exception = future.result(), future.exception()

        # if exception is not None:
        #     raise exception

        outputs = []
        for output in [
            "output_fpn_efficientnetb5",
            "output_unet_efficientnetb5",
            "output_linknet_efficientnetb5",
        ]:
            tensor_shape = [
                    int(v)
                    for v in str(result.outputs[output].tensor_shape.dim)
                    .replace("[", "")
                    .replace("]", "")
                    .replace("size: ", "")
                    .split(", ")
            ]

            batch_out = np.array(result.outputs[output].float_val).reshape(tensor_shape)
            outputs.append(batch_out)

        # Concatenate the predictions from the 3 models along the channel axis
        outputs = np.concatenate(outputs, axis=0)
        # Aggregate the predictions from the 3 models along the channel axis
        outputs = np.mean(outputs, axis=0)

        pred = np.expand_dims(cv2.resize(outputs,(width, height), 
            interpolation=cv2.INTER_CUBIC),axis=-1)

        # crf 
        pred_chw = np.transpose(pred, (2, 0, 1))  # HWC -> CHW
        pred_chw = np.concatenate([1 - pred_chw, pred_chw], axis=0)

        out = dense_crf(image, pred_chw)

        out = np.argmax(out, axis=0)
        pred = out

        # # binarization threshold
        pred = (pred * 255).astype("uint8")
        _, pred = cv2.threshold(
            pred, int(self.threshold * 255), 255, cv2.THRESH_OTSU)
        pred = pred / np.float32(255)

        # post processing
        pred = postprocess(pred)

        mask = (pred * 255).astype("uint8")

        # output_masks = tensor_util.MakeNdarray(
        #     result.outputs[self.output_name])

        # num_images, height, width, channels = images.shape
        # output_masks = output_masks.reshape(num_images, height, width, 1)

        return mask

    def aspect_aware_resizing(
        self, image, max_size, interpolation=Image.LANCZOS):
        """Performs resizing while maintaining the aspect ratio of the image

        Parameters
        ----------
        image : instance of `numpy.ndarray`
            Image matrix, which shoud be in the specified mode.
        max_size : int
            The maximum allowed image size, which applies to both width and
            height.

            The larger of the 2 will be used to compute the ratio of
            downsampling, which is then applied to both dimensions.
        interpolation : int, optional, defaults to PIL.Image.LANCZOS
            Interpolation algorithm using during downsampling, which should be
            one of the supported enums for `PIL.Image.resize`.

        Returns
        -------
        image : instance of `numpy.ndarray`
            Image with dimensions guaranteed to be within the bounds specified
            by the `max_size` parameter.

        """

        height, width = image.shape[:2]
        larger_of_the_two = max(height, width)

        if larger_of_the_two > max_size:
            scaling_value = max_size / float(larger_of_the_two)

            resize_height = int(np.floor(height * scaling_value))
            resize_width = int(np.floor(width * scaling_value))

            pillow_image = Image.fromarray(image)
            pillow_image = pillow_image.resize(
                (resize_width, resize_height),
                resample=interpolation)
            image = np.asarray(pillow_image)

        return image
