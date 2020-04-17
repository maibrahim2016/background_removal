import sys

if sys.version_info.major == 2:
    import mock
else:
    from unittest import mock

from nose.tools import assert_equal
from nose.tools import assert_is_instance
from nose.tools import assert_true
from nose.tools import assert_tuple_equal

from .. import exceptions

def test_CodedException_init():
    mock_status_code = mock.MagicMock(name='mock status code')
    exception = exceptions.CodedException(mock_status_code)

    expected_attributes_and_values = [
        ('status_code', mock_status_code),
    ]
    for (attribute, value) in expected_attributes_and_values:
        assert_true(hasattr(exception, attribute),
            msg='expected exception to have `{0}` attribute'.format(
                attribute))
        assert_equal(getattr(exception, attribute), value,
            msg='incorrect value for `{0}` attribute'.format(
                attribute))

def test_CodedException_heritage():
    mock_status_code = mock.MagicMock(name='mock status code')
    exception = exceptions.CodedException(mock_status_code)

    assert_is_instance(exception, RuntimeError,
        msg='expected `CodedException` to inherit from RuntimeError')

def test_CodedException_init_with_arguments_for_superclass():
    mock_status_code = mock.MagicMock(name='mock status code')
    additional_args = [
        mock.MagicMock(name='mock 1st argument'),
        mock.MagicMock(name='mock 2nd argument'),
        mock.MagicMock(name='mock 3rd argument'),
    ]
    
    exception = exceptions.CodedException(
        mock_status_code,
        additional_args[0],
        additional_args[1],
        additional_args[2])

    assert_tuple_equal(exception.args, tuple(additional_args),
        msg=('expected the additional arguments provided to be contained '
             'within `args` attribute'))
