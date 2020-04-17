
class CodedException(RuntimeError):
    """Exception with HTTP status codes

    This exception class is built mainly as a convenient means of creating and
    subsequently raising errors within the microservice, whereby the response
    status code will be assigned according to the raised exception, if such an
    attribute exists.

    Parameters
    ----------
    status_code : int
        HTTP status code corresponding to the kind of error you would like to
        raise.
    *args
        Additional positional arguments are passed to the parent class.
    *kwargs
        Additional keyworded arguments are passed to the parent class.

    Attributes
    ----------
    status_code : int
        Refer to `status_code` parameter.

    """

    def __init__(self, status_code, *args, **kwargs):
        RuntimeError.__init__(self, *args, **kwargs)
        self.status_code = status_code
