import sys

if sys.version_info.major == 2:
    import httplib
else:
    from http import client as httplib

import argparse
import cv2
import json
import numpy as np
import skimage.color
import skimage.io
import traceback

from models.segmenter import Segmenter
from sepyroth.daemons.daemons import HttpDaemon
from sepyroth.daemons.http_ext import (BaseDaemonHttpRequestHandler,
                                       ForkedHttpServer,
                                       ThreadedHttpServer)
from sepyroth.utils.config_helper import from_json
from six.moves import zip
from utils.exceptions import CodedException
from utils.misc_utils import base64_to_image, image_to_base64

import segmentation_models as sm

class SegmenterHttpDaemon(HttpDaemon):
    """HTTP daemon implementation for the segmentation service

    Attributes
    ----------
    model : instance of `models.segmenter.Segmenter`
        Initialised proxy model.

    """

    def initialise(self):
        """Initialises the proxy model
        
        """

        self.logger.info('initialising proxy model')
        self.model = Segmenter(
            self.configurations.model_server_host,
            self.configurations.model_server_port,
            self.configurations.model_name,
            self.configurations.signature_name,
            self.configurations.input_name,
            self.configurations.output_name,
            self.configurations.threshold,
            self.configurations.request_timeout,
            self.configurations.max_send_message_length,
            self.configurations.max_receive_message_length)

    def __call__(self):
        try:
            self.logger.info(
                'starting up server at port {port}'.format(port=self.port))
            HttpDaemon.__call__(self)
            self.logger.info('shutting down server')
        except:
            self.logger.critical('abnormal termination\n{traceback}'.format(
                traceback=traceback.format_exc()))

    def required_configurations(self):
        return HttpDaemon.required_configurations(self) + \
            [
                # required model parameters
                'input_name',
                'max_receive_message_length',
                'max_send_message_length',
                'model_name',
                'model_server_host',
                'model_server_port',
                'output_name',
                'request_timeout',
                'signature_name',
                'patch_size',
                'threshold'
            ]

    def stop_running(self, *args):
        """Registered for signal handling

        We override this to allow for additional arguments to be specified to
        the signal handler, necessary in the case of using systemd.

        This wasn't a problem in the past, but was actually an overlooked
        requirement when building the daemonisation module, as per Python's
        documentation on how to build signal handlers.

        """

        HttpDaemon.stop_running(self)

class SegmenterHttpHandler(BaseDaemonHttpRequestHandler):

    def do_GET(self):
        self.send_response(httplib.OK)
        self.end_headers()
        self.wfile.write(
            'Hello, {class_name} up and running'.format(
                class_name=self.__class__.__name__).encode('utf-8'))
        return

    def do_POST(self):
        output = {
            'ok': False
        }

        try:
            params = self.parse_params()

            if 'images' not in params:
                raise CodedException(
                    httplib.BAD_REQUEST,
                    'missing required parameter `images`')

            base64_images = json.loads(params['images'])

            if len(base64_images) == 0:
                raise CodedException(
                    httplib.BAD_REQUEST,
                    'parameter `images` must contain at least 1 value')

            output_masks = self._infer(base64_images)

            response_code = httplib.OK
            output.update({
                'masks': json.dumps(output_masks),
                'ok': True,
            })

        except Exception as exception:
            output.update({
                'error_message': str(exception),
            })
            response_code = getattr(
                exception,
                'status_code',
                httplib.INTERNAL_SERVER_ERROR)

            self.daemon.logger.exception('encountered exception')
        
        self.send_response(response_code)
        self.end_headers()
        self.wfile.write(json.dumps(output).encode('utf-8'))
        return

    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        BaseDaemonHttpRequestHandler.end_headers(self)

    def _infer(self, base64_images):
        """Performs inference on the provided base64-encoded images

        The images are decoded into numpy arrays, packaged, sent for inference
        and then the resulting masks are re-encoded into their base64 forms.

        If the image's width or height exceeds the configured maximum size, it
        will be resized to said maximum size, while retaining its aspect ratio.

        Based on Brian's provided implementation, the masks are to be resized
        to the desired dimensions, then the same rule is applied i.e. all cells
        with value above 0 are set to a "true" indicator (which is based on the
        microservice configuration).

        Parameters
        ----------
        base64_images : list
            The list of base64-encoded images which will be decoded into numpy
            arrays and then packaged for inference at the model server.

        Returns
        -------
        output_masks : list
            The list of base64-encoded masks produced by the model.

        """
        
        images = [cv2.cvtColor(base64_to_image(base64_image.encode('ascii')), cv2.COLOR_BGR2RGB)
                        for base64_image in base64_images]
        dimensions = [image.shape[:2] for image in images]

        output_masks = []
        for image, (height, width) in zip(images, dimensions):
            resized_image = cv2.resize(image, (self.daemon.configurations.patch_size,self.daemon.configurations.patch_size))
            preprocessing_fn = sm.get_preprocessing("efficientnetb5")
            test_image_preprocessed = preprocessing_fn(resized_image)
            images_tensor = np.expand_dims(test_image_preprocessed, axis=0)
            mask = self.daemon.model(image,images_tensor,height,width)
            # remove unnecessary first axis
            # mask = np.squeeze(mask[0])
            # mask = cv2.resize(mask, (width, height))

            output_masks.append(mask)

        output_masks = [image_to_base64(mask) for mask in output_masks]

        return output_masks

        # max_image_size = self.daemon.configurations.max_image_size
        # interpolation = self.daemon.configurations.resizing_interpolation
        # mask_interpolation = self.daemon.configurations.mask_interpolation
        # mask_true_value = self.daemon.configurations.mask_true_value

        # # this is currently a hardcoded requirement that prevents images from
        # # being processed in batches, specified by Brian
        # #
        # # if in some distant time in the future, a decision is made to support
        # # batch processing, you'll have to determine the desired approach for
        # # reshaping the images to the same dimensions

        
        # for image, (height, width) in zip(images, dimensions):

        #     resized_image = self.daemon.model.aspect_aware_resizing(
        #         image,
        #         max_image_size,
        #         interpolation)

        #     mask = self.daemon.model(
        #         np.expand_dims(resized_image, axis=0))

        #     # remove unnecessary first axis
        #     mask = np.squeeze(mask, axis=0)

        #     # resize the mask then convert all non-zero values to the
        #     # configured "true" value, then set the data type to unsigned int8
        #     mask = cv2.resize(
        #         mask.astype(np.float32),
        #         (width, height),
        #         interpolation=mask_interpolation)
        #     mask[mask > 0] = mask_true_value
        #     mask = mask.astype(np.uint8)

        #     output_masks.append(mask)

        # output_masks = [image_to_base64(mask) for mask in output_masks]

        # return output_masks

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'operation',
        help='daemon operation to execute [start | stop]',
        choices=['start', 'stop'])
    parser.add_argument(
        '--json',
        required=True,
        help='path to JSON configuration file')
    parser.add_argument(
        '--server',
        default='threaded',
        help='HTTP server forking strategy [forked | threaded]',
        choices=['forked', 'threaded'])
    args = parser.parse_args()

    configurations = from_json(args.json)

    if args.server == 'threaded':
        server_cls = ThreadedHttpServer
    else:
        server_cls = ForkedHttpServer

    daemon = SegmenterHttpDaemon(
        configurations,
        handler_cls=SegmenterHttpHandler,
        server_cls=server_cls)
    daemon.execute_operation(args.operation)
