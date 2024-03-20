import enum
import warnings

import numpy

from .config.core import Config
from . import classifiers
from .model_io import OutputMode, InputEncoding



class InputConversion(enum.Enum):
    Concatenate = enum.auto()
    OneHotAsInteger = enum.auto()
    VectorAsBinary = enum.auto()
    ComposeAsMatrix = enum.auto()

    def get_input_encoding(self):
        match self:
            case self.Concatenate:
                return InputEncoding.Vector
            case self.OneHotAsInteger:
                return InputEncoding.Vector
            case self.VectorAsBinary:
                return InputEncoding.Vector
            case self.ComposeAsMatrix:
                return InputEncoding.Matrix

    def to_json(self):
        match self:
            case self.Concatenate:
                return 'concatenate'
            case self.OneHotAsInteger:
                return 'one-hot-as-integer'
            case self.VectorAsBinary:
                return 'vector-as-binary'
            case self.ComposeAsMatrix:
                return 'compose-as-matrix'

    @classmethod
    def from_json(cls, string: str):
        match string:
            case 'concatenate':
                return cls.Concatenate
            case 'one-hot-as-integer':
                return cls.OneHotAsInteger
            case 'vector-as-binary':
                return cls.VectorAsBinary
            case 'compose-as-matrix':
                return cls.ComposeAsMatrix


def build_stacking_classifier(conf: Config):
    # First, we load the settings
    model_name = conf.get('run.stacking-meta-classifier')
    # Load hyper params
    model_params_raw = conf.get('run.stacking-meta-classifier-hyper-parameters')
    assert len(model_params_raw) == 1
    key = list(model_params_raw)[0]
    model_params = model_params_raw[key][0]

    must_concat = conf.get('run.stacking-use-concat')
    no_matrix = conf.get('run.stacking-no-matrix')
    model_factory = classifiers.models[model_name]
    output_mode = OutputMode.from_string(conf.get('run.output_mode'))

    # Determine the input encoding.
    # For detection, the input encoding is an integer.
    # For classification, we ideally want a matrix as the
    # input encoding. However, we coerce to a numerical
    # vector of one-hot indices if this is not supported.
    # For 3-classification, we view the three classifications
    # combined as a binary number and use this to
    # get a single number.
    # Alternatively, if stacking-use-concat is given, we simply
    # concatenate all predictions, resulting in a vector output.
    warnings.warn('Be aware of input conversion methods used in stacking')
    if must_concat:
        input_conversion = InputConversion.Concatenate
    else:
        supports_matrix = InputEncoding.Matrix in model_factory.supported_input_encodings()
        conversion_methods = {
            # input encoding, matrix allowed
            (OutputMode.Detection, False): InputConversion.Concatenate,
            (OutputMode.Detection, True): InputConversion.Concatenate,
            (OutputMode.Classification8, False): InputConversion.OneHotAsInteger,
            (OutputMode.Classification8, True): InputConversion.ComposeAsMatrix,
            (OutputMode.Classification3, False): InputConversion.VectorAsBinary,
            (OutputMode.Classification3, True): InputConversion.ComposeAsMatrix,
            (OutputMode.Classification3Simplified, False): InputConversion.OneHotAsInteger,
            (OutputMode.Classification3Simplified, True): InputConversion.ComposeAsMatrix
        }
        matrix_allowed = supports_matrix and not no_matrix
        input_conversion = conversion_methods[(output_mode, matrix_allowed)]

    # Now, determine the size of the input.
    number_of_classifiers = len(conf.get('run.classifier'))
    match input_conversion:
        case InputConversion.Concatenate:
            input_size = number_of_classifiers * output_mode.output_size
        case InputConversion.VectorAsBinary:
            input_size = number_of_classifiers
        case InputConversion.OneHotAsInteger:
            input_size = number_of_classifiers
        case InputConversion.ComposeAsMatrix:
            input_size = (number_of_classifiers, output_mode.output_size)
        case _:
            raise NotImplementedError

    model: classifiers.AbstractModel = model_factory(
        input_size,
        input_conversion.get_input_encoding(),
        output_mode.output_size,
        output_mode.output_encoding
    )

    def factory():
        return model.get_compiled_model(**model_params)

    return factory, input_conversion


def transform_predictions_to_stacking_input(output_mode: OutputMode,
                                            predictions,
                                            input_conversion: InputConversion):
    match input_conversion:
        case InputConversion.Concatenate:
            converted = []
            for prediction in predictions:
                converted.append(prediction.ravel())
            return numpy.vstack(converted).transpose()
        case InputConversion.OneHotAsInteger:
            converted = []
            for prediction in predictions:
                converted.append(numpy.argmax(prediction, axis=1))
            return numpy.vstack(converted).transpose()
        case InputConversion.VectorAsBinary:
            number_of_digits = output_mode.output_size
            single_mask = [2**(number_of_digits - i - 1) for i in range(number_of_digits)]
            mask = numpy.asarray([
                single_mask.copy()
                for _ in range(len(predictions))
            ])
            converted = [
                numpy.multiply(mask, prediction_set).sum(axis=1)
                for prediction_set in predictions
            ]
            return numpy.vstack(converted).transpose()
        case InputConversion.ComposeAsMatrix:
            return numpy.stack(predictions, axis=-2)

