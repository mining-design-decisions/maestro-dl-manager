from .config import Argument
from .config import IntArgument
from .config import FloatArgument
from .config import BoolArgument
from .config import StringArgument
from .config import NestedArgument
from .config import EnumArgument


def get_metrics_endpoint_data():
    return {
        'name': 'metrics',
        'help': 'Endpoint to calculate various metrics based on predictions.',
        'private': False,
        'args': _get_metrics_args()
    }


def get_confusion_matrix_endpoint_data():
    return {
        'name': 'confusion-matrix',
        'description': 'Endpoint to calculate the confusion matrix (or matrices) for a given training task.',
        'private': False,
        'args': _get_metric_base_args()
    }


def _get_database_args() -> dict[str, Argument]:
    return {
        'database-url': StringArgument(
            name='database-url',
            description='URL of the database (wrapper)'
        )
    }


def _get_metric_base_args() -> dict[str, Argument]:
    return _get_database_args() | {
        'model-id': StringArgument(
            name='model-id',
            description='ID of the model from which predictions must be fetched'
        ),
        'version-id': StringArgument(
            name='version-id',
            description='ID of the model version from which predictions must be fetched'
        ),
        'classification-as-detection': BoolArgument(
            name='classification-as-detection',
            description='Evaluate detection performance of a classification model',
            default=False
        ),
        'epoch': StringArgument(
            name='epoch',
            description='Epoch to evaluate metrics at. Either an epoch, `last`, `stopping-point`, or `all`'
        ),
        'include-non-arch': BoolArgument(
            name='include-non-arch',
            description='Include the non-architectural class as a class in Classification3',
            default=False
        )
    }


def _get_metrics_args() -> dict[str, Argument]:
    return _get_metric_base_args() | {
        'metrics': ...
    }
