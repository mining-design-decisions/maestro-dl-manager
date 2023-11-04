from . import classifiers
from . import embeddings
from . import feature_generators
from . import upsampling

from .config.arguments import Argument
from .config.arguments import IntArgument
from .config.arguments import FloatArgument
from .config.arguments import QueryArgument
from .config.arguments import BoolArgument
from .config.arguments import StringArgument
from .config.arguments import NestedArgument
from .config.arguments import EnumArgument, DynamicEnumArgument
from .config.arguments import JSONArgument
from .config.arguments import ListArgument
from .config.constraints import Constraint
from .config.constraints import BooleanConstraint
from .config.constraints import Forbids
from .config.constraints import MutuallyExclusive
from .config.constraints import Or
from .config.constraints import Equal
from .config.constraints import NotEqual
from .config.constraints import ListContains
from .config.constraints import ListNotContains
from .config.constraints import Constant
from .config.constraints import ArgumentRef
from .config.constraints import LengthOfArgument
from .config import schemas

##############################################################################
##############################################################################
# Top-level access function
##############################################################################


def get_api_spec():
    return {
        'name': 'Maestro Deep Learning Manager',
        'help': 'A Deep Learning Backend for Maestro.',
        'commands': {
            'train': get_train_endpoint_data(),
            'run': get_run_endpoint_data(),
            'predict': get_prediction_endpoint_data(),
            'generate-embedding': get_embedding_endpoint_data(),
            'generate-embedding-internal': get_internal_embedding_endpoint_data(),
            'metrics': get_metrics_endpoint_data(),
            'confusion-matrix': get_confusion_matrix_endpoint_data()
        }
    }


##############################################################################
##############################################################################
# Shared arguments
##############################################################################


def _get_database_args() -> dict[str, Argument]:
    return {
        'database-url': StringArgument(
            name='database-url',
            description='URL of the database (wrapper)'
        )
    }


##############################################################################
##############################################################################
# Training arguments
##############################################################################

def get_train_endpoint_data():
    return {
        'name': 'train',
        'help': 'Variant of the `run` command which loads a config from the database.',
        'private': False,
        'args': _get_train_endpoint_args(),
        'constraints': _get_train_endpoint_constraints()
    }

def _get_train_endpoint_constraints() -> list[Constraint]:
    return []

def _get_train_endpoint_args() -> dict[str, Argument]:
    return _get_database_args() | {
        'model-id': StringArgument(
            name='model-id',
            description='ID of the model config to train'
        )
    }

def get_run_endpoint_data():
    return {
        'name': 'run',
        'help': 'Train a classifier and store the results',
        'private': True,
        'args': _get_run_endpoint_args(),
        'constraints': _get_run_endpoint_constraints()
    }

def _get_run_endpoint_constraints() -> list[Constraint]:
    return [
        BooleanConstraint(
            Equal(LengthOfArgument('classifier'), LengthOfArgument('input-mode')),
            message=f'Argument lists "classifier" and "input-mode" must have equal length.'
        ),
        BooleanConstraint(
            Equal(LengthOfArgument('early-stopping-min-delta'), LengthOfArgument('early-stopping-attribute')),
            message=f'Argument lists "early-stopping-min-delta" and "early-stopping-attribute" must have equal length.'
        ),
        MutuallyExclusive(
            Equal(ArgumentRef('ensemble-strategy'), Constant('none')),
            Equal(ArgumentRef('test-separately'), Constant(True)),
            message='Cannot use ensemble when using separate testing mode.'
        ),
        MutuallyExclusive(
            Equal(ArgumentRef('store-model'), Constant(True)),
            Equal(ArgumentRef('test-separately'), Constant(True)),
            message='Cannot store model when using separate testing mode.'
        ),
        MutuallyExclusive(
            Equal(ArgumentRef('store-model'), Constant(True)),
            NotEqual(ArgumentRef('k-cross'), Constant(0)),
            Equal(ArgumentRef('cross-project'), Constant(True)),
            message='Cannot store model when using k-fold cross validation or project cross validation.'
        ),
        MutuallyExclusive(
            Equal(ArgumentRef('cross-project'), Constant(True)),
            NotEqual(ArgumentRef('k-cross'), Constant(0)),
            message='Cannot use k-cross and cross-project at the same time.'
        ),
        Forbids(
            main=Equal(ArgumentRef('quick-cross'), Constant(True)),
            message='Must specify k-cross when running with quick-cross.',
            forbids=[
                Equal(ArgumentRef('k-cross'), Constant(0)),
            ]
        ),
        Forbids(
            main=Equal(ArgumentRef('store-model'), Constant(True)),
            message='model-id must be given when storing a model.',
            forbids=[
                Equal(ArgumentRef('model-id'), Constant(''))
            ]
        ),
        Forbids(
            main=Equal(ArgumentRef('store-model'), Constant(True)),
            message='May not use cache-features when using store-model.',
            forbids=[
                Equal(ArgumentRef('cache-features'), Constant(True))
            ],
            add_reverse_constraints=True
        ),
        Forbids(
            main=Equal(ArgumentRef('analyze-keywords'), Constant(True)),
            message='Can only analyze keywords when using a convolutional model',
            forbids=[
                ListNotContains(ArgumentRef('classifier'), Constant('LinearConv1Model'))
            ],
            add_reverse_constraints=True
        ),
        Forbids(
            main=Equal(ArgumentRef('analyze-keywords'), Constant(True)),
            message='Cannot perform cross validation when extracting keywords.',
            forbids=[
                NotEqual(ArgumentRef('k-cross'), Constant(0)),
                Equal(ArgumentRef('cross-project'), Constant(True))
            ],
            add_reverse_constraints=True
        ),
        Forbids(
            main=Equal(ArgumentRef('k-cross'), Constant(True)),
            message='Must test with training data when performing cross validation!',
            forbids=[
                Equal(ArgumentRef('test-with-training-data'), Constant(False))
            ],
            add_reverse_constraints=True
        ),
        Forbids(
            main=Equal(ArgumentRef('cross-project'), Constant(True)),
            message='Must test with training data when performing cross validation!',
            forbids=[
                Equal(ArgumentRef('test-with-training-data'), Constant(False))
            ],
            add_reverse_constraints=True
        ),
        Forbids(
            main=Or(
                Equal(ArgumentRef('apply-ontology-classes'), Constant(True)),
                ListContains(ArgumentRef('classifier'), Constant('OntologyFeatures'))
            ),
            message='Ontology class file must be given when applying ontology classes or using ontology features.',
            forbids=[
                Equal(ArgumentRef('ontology-classes'), Constant(''))
            ]
        )
    ]

def _get_run_endpoint_args():
    return _get_database_args() | {
        'input-mode': ListArgument(
            inner=DynamicEnumArgument(
                name='input-mode',
                description='Feature generator to use.',
                lookup_map=feature_generators.generators
            )
        ),
        'output-mode': EnumArgument(
            name='output-mode',
            description='Output mode (classification task) to use.',
            options=[
                "Detection",
                "Classification3Simplified",
                "Classification3",
                "Classification8"
            ]
        ),
        'params': NestedArgument(
            name='params',
            description='Feature generator parameters',
            spec={
                name: value.get_arguments()
                for name, value in feature_generators.generators.items()
            },
            multi_valued=True
        ),
        'ontology-classes': StringArgument(
            name='ontology-classes',
            description='ID of the DB-file containing ontology classes.',
            default=''
        ),
        'apply-ontology-classes': BoolArgument(
            name='apply-ontology-classes',
            description='Enable application of ontology classes.',
            default=False
        ),
        'classifier': ListArgument(
            inner=DynamicEnumArgument(
                name='classifier',
                description='Classifier to use.',
                lookup_map=classifiers.models
            )
        ),
        'epochs': IntArgument(
            name='epochs',
            description='Amount of training epochs'
        ),
        'split-size': FloatArgument(
            name='split-size',
            description='Size of testing and validation splits. Ignored when using cross validation',
            default=0.2
        ),
        'max-train': IntArgument(
            name='max-train',
            description='Maximum amount of training items. -1 for infinite.',
            default=-1
        ),
        'k-cross': IntArgument(
            name='k-cross',
            description='Enable k-fold cross-validation (k > 0).',
            default=0
        ),
        'quick-cross': BoolArgument(
            name='quick-cross',
            description='Enable quick (non-nested) cross validation.',
            default=False
        ),
        'cross-project': BoolArgument(
            name='cross-project',
            description='Enable project based cross validation.',
            default=False
        ),
        'cache-features': BoolArgument(
            name='cache-features',
            description='Force caching of features. '
                        'Note that the deep learning manager does not handle cache invalidation.',
            default=False
        ),
        'architectural-only': BoolArgument(
            name='architectural-only',
            description='Restrict training set to only architectural issues.',
            default=False
        ),
        'hyper-params': NestedArgument(
            name='hyper-parameters',
            description='Hyper-parameters for the classifiers',
            spec={
                name: value.get_arguments()
                for name, value in classifiers.models.items()
            },
            multi_valued=True
        ),
        'class-balancer': EnumArgument(
            name='class-balancer',
            description='Class balancing algorithm to use.',
            options=['none', 'upsample', 'class-weights'],
            default='none'
        ),
        'upsampler': DynamicEnumArgument(
            name='upsampler',
            description='Upsampling method to use',
            lookup_map=upsampling.upsamplers,
            enabled_if=Equal(
                ArgumentRef('class-balancer'), Constant('upsample')
            )
        ),
        'upsampler-params': NestedArgument(
            name='upsampler-params',
            description='Parameters for the upsampler',
            spec={
                name: value.get_arguments()
                for name, value in upsampling.upsamplers.items()
            },
            multi_valued=False,
            enabled_if=Equal(
                ArgumentRef('class-balancer'), Constant('upsample')
            )
        ),
        'batch-size': IntArgument(
            name='batch-size',
            description='Specify the batch size used during training',
            default=32
        ),
        'combination-strategy': EnumArgument(
            name='combination-strategy',
            description='Strategy to combine models into a single network.',
            options=[
                "add",
                "subtract",
                "average",
                "min",
                "max",
                "multiply",
                "dot",
                "concat"
            ],
            enabled_if=Equal(
                ArgumentRef('ensemble-strategy'), Constant('combination')
            )
        ),
        'combination-model-hyper-params': NestedArgument(
            name='combination-model-hyper-params',
            description='Hyper-parameters for the creation of a combined model '
                        '(mainly optimiser and learning rate.',
            spec={
                name: value.get_arguments()
                for name, value in classifiers.models.items()
            },
            multi_valued=False,
            enabled_if=Equal(
                ArgumentRef('ensemble-strategy'), Constant('combination')
            )
        ),
        'ensemble-strategy': EnumArgument(
            name='ensemble-strategy',
            description='Strategy used to combine multiple models.',
            default='none',
            options=['stacking', 'voting', 'combination', 'none']
        ),
        'stacking-meta-classifier': DynamicEnumArgument(
            name='stacking-meta-classifier',
            description='Classifier to use as meta-classifier in stacking.',
            lookup_map=classifiers.models,
            enabled_if=Equal(
                ArgumentRef('ensemble-strategy'), Constant('stacking')
            )
        ),
        'stacking-meta-classifier-hyper-parameters': NestedArgument(
            name='stacking-meta-classifier-hyper-parameters',
            description='Hyper-parameters for the meta classifier.',
            spec={
                name: value.get_arguments()
                for name, value in classifiers.models.items()
            },
            multi_valued=False,
            enabled_if=Equal(
                ArgumentRef('ensemble-strategy'), Constant('stacking')
            )
        ),
        'stacking-use-concat': BoolArgument(
            name='stacking-use-concat',
            description='Use simple concatenation to create the input for the meta classifier',
            default=False,
            enabled_if=Equal(
                ArgumentRef('ensemble-strategy'), Constant('stacking')
            )
        ),
        'stacking-no-matrix': BoolArgument(
            name='stacking-no-matrix',
            description='Disallow the use of matrices for meta classifier input.',
            default=False,
            enabled_if=Equal(
                ArgumentRef('ensemble-strategy'), Constant('stacking')
            )
        ),
        'voting-mode': EnumArgument(
            name='voting-mode',
            description='Mode for the voting ensemble. Either hard of sort voting',
            options=['soft', 'hard'],
            enabled_if=Equal(
                ArgumentRef('ensemble-strategy'), Constant('voting')
            )
        ),
        'use-early-stopping': BoolArgument(
            name='use-early-stopping',
            description='Enable early stopping',
            default=False
        ),
        'early-stopping-patience': IntArgument(
            name='early-stopping-patience',
            description='Patience used when using early stopping.',
            default=5
        ),
        'early-stopping-min-delta': ListArgument(
            inner=FloatArgument(
                name='early-stopping-min-delta',
                description='Minimum delta used when using early stopping. '
                            'One entry for every attribute used, '
                            'or one shared delta for all attributes.',
                default=0.001
            ),
            default=[0.001]
        ),
        'early-stopping-attribute': ListArgument(
            inner=StringArgument(
                name='early-stopping-attribute',
                description='Attribute(s) to use for early stopping (from the validation set)',
                default='loss'
            ),
            default=['loss']
        ),
        'test-separately': BoolArgument(
            name='test-separately',
            description='If given, disable ensemble mode. '
                        'Instead, all classifiers are trained and evaluated sequentially.',
            default=False
        ),
        'store-model': BoolArgument(
            name='store-model',
            description='If given, store the trained model. Incompatible with cross validation.',
            default=False
        ),
        'model-id': StringArgument(
            name='model-id',
            description='ID of the model (config) being trained. '
                        'Does not need to be passed as argument '
                        '(automatically inserted by /train endpoint)',
        ),
        'analyze-keywords': BoolArgument(
            name='analyze-keywords',
            description='Compute list of important keywords based on '
                        'classifier predictions (convolutional model only).',
            default=False
        ),
        'training-data-query': QueryArgument(
            name='training-data-query',
            description='Query to fetch training data'
        ),
        'test-data-query': QueryArgument(
            name='test-data-query',
            description='Query to fetch testing data. '
                        'Only necessary when not using a normal train/test split.',
            enabled_if=Equal(
                ArgumentRef('test-with-training-data'), Constant(False)
            ),
        ),
        'test-with-training-data': BoolArgument(
            name='test-with-training-data',
            description='Draw testing data from training data using train/test split.',
            default=True
        ),
        'seed': IntArgument(
            name='seed',
            description='Seed used to make training deterministic. Use -1 to disable use of a seed.',
            default=-1
        ),
        'perform-tuning': BoolArgument(
            name='perform-tuning',
            description='Enable hyperparameter tuning.',
            default=False
        ),
        'tuner-type': EnumArgument(
            name='tuner-type',
            description='Select the hyperparameter optimization strategy.',
            options=['RandomSearch', 'BayesianOptimization', 'Hyperband'],
            enabled_if=Equal(
                ArgumentRef('perform-tuning'), Constant(True)
            )
        ),
        'tuner-objective': StringArgument(
            name='tuner-objective',
            description='Metric/objective to optimise while tuning.',
            enabled_if=Equal(
                ArgumentRef('perform-tuning'), Constant(True)
            )
        ),
        'tuner-max-trials': IntArgument(
            name='tuner-max-trials',
            description='Select the number of hyperparameter combinations that are tried.',
            enabled_if=Equal(
                ArgumentRef('perform-tuning'), Constant(True)
            )
        ),
        'tuner-executions-per-trial': IntArgument(
            name='tuner-executions-per-trial',
            description='Select the number of executions per trial, to mitigate randomness.',
            enabled_if=Equal(
                ArgumentRef('perform-tuning'), Constant(True)
            )
        ),
        'tuner-hyperband-iterations': IntArgument(
            name='tuner-hyperband-iterations',
            description='Select the number of iterations for the HyperBand algorithm.',
            enabled_if=Equal(
                ArgumentRef('perform-tuning'), Constant(True)
            )
        ),
        'tuner-hyper-params': NestedArgument(
            name='tuner-hyper-params',
            description='Hyper-parameter search space when tuning models.',
            spec={
                name: value.get_arguments()
                for name, value in classifiers.models.items()
            },
            tunable=True,
            multi_valued=True,
            enabled_if=Equal(
                ArgumentRef('perform-tuning'), Constant(True)
            )
        ),
        'tuner-combination-model-hyper-params': NestedArgument(
            name='tuner-combination-model-hyper-params',
            description='Hyper-parameters for the creation of a combined model for keras tuner.',
            spec={
                name: value.get_arguments()
                for name, value in classifiers.models.items()
            },
            tunable=True,
            multi_valued=False,
            enabled_if=Equal(
                ArgumentRef('perform-tuning'), Constant(True)
            )
        )
    }


##############################################################################
##############################################################################
# Prediction arguments
##############################################################################


def get_prediction_endpoint_data():
    return {
        'name': 'predict',
        'help': 'Use an existing classifier to make predictions on new data.',
        'private': False,
        'args': _get_prediction_args(),
        'constraints': _get_prediction_endpoint_constraints()
    }


def _get_prediction_endpoint_constraints() -> list[Constraint]:
    return []


def _get_prediction_args() -> dict[str, Argument]:
    return _get_database_args() | {
        'model': StringArgument(
            name='model',
            description='ID of the model to predict with'
        ),
        'version': StringArgument(
            name='version',
            description='ID of the version of the model to predict with. '
                        'Use `most-recent` for most recently trained version.',
            default='most-recent'
        ),
        'data-query': QueryArgument(
            name='data-query',
            description='Query used to retrieve data to compute predictions for.'
        )
    }


##############################################################################
##############################################################################
# Embedding arguments
##############################################################################


def get_embedding_endpoint_data():
    return {
        'name': 'generate-embedding',
        'help': 'Generate a word or document embedding for use in a feature generator.',
        'private': False,
        'args': _get_embedding_endpoint_args(),
        'constraints': _get_embedding_endpoint_constraints()
    }


def _get_embedding_endpoint_constraints() -> list[Constraint]:
    return []


def _get_embedding_endpoint_args() -> dict[str, Argument]:
    return _get_shared_embedding_args()

def get_internal_embedding_endpoint_data():
    return {
        'name': 'generate-embedding-internal',
        'help': 'Internal implementation endpoint for generate-embedding endpoint.',
        'private': True,
        'args': _get_internal_embedding_endpoint_args(),
        'constraints': _get_internal_embedding_endpoint_constraints()
    }


def _get_internal_embedding_endpoint_constraints() -> list[Constraint]:
    return []


def _get_internal_embedding_endpoint_args() -> dict[str, Argument]:
    return _get_shared_embedding_args() | {
        'embedding-generator': DynamicEnumArgument(
            name='embedding-generator',
            description='Type of embedding to train',
            lookup_map=embeddings.generators
        ),
        'embedding-config': NestedArgument(
            name='embedding-config',
            description='Config of the embedding.',
            spec={
                name: value.get_arguments()
                for name, value in embeddings.generators.items()
            },
            multi_valued=False
        ),
        'training-data-query': QueryArgument(
            name='training-data-query',
            description='Query to obtain data from the database for training.'
        )
    }


def _get_shared_embedding_args() -> dict[str, Argument]:
    return _get_database_args() | {
        'embedding-id': StringArgument(
            name='embedding-id',
            description='ID of the embedding in the database to train.'
        )
    }

##############################################################################
##############################################################################
# Metric endpoint arguments
##############################################################################


def get_metrics_endpoint_data():
    return {
        'name': 'metrics',
        'help': 'Endpoint to calculate various metrics based on predictions.',
        'private': False,
        'args': _get_metrics_endpoint_args(),
        'constraints': _get_metrics_endpoint_constraints()
    }

def _get_metrics_endpoint_constraints() -> list[Constraint]:
    return []

def _get_metrics_endpoint_args() -> dict[str, Argument]:
    # TODO: further refine schema
    return _get_metric_base_args() | {
        'metrics': JSONArgument(
            name='metrics',
            description='JSON description of the metrics to compute',
            schema=schemas.Array(
                schemas.FixedObject(
                    dataset=schemas.String(),
                    metric=schemas.String(),
                    variant=schemas.String()
                )
            )
        )
    }


def get_confusion_matrix_endpoint_data():
    return {
        'name': 'confusion-matrix',
        'help': 'Endpoint to calculate the confusion matrix (or matrices) for a given training task.',
        'private': False,
        'args': _get_confusion_matrix_endpoint_args(),
        'constraints': _get_confusion_matrix_endpoint_constraints()
    }


def _get_confusion_matrix_endpoint_constraints() -> list[Constraint]:
    return []


def _get_confusion_matrix_endpoint_args() -> dict[str, Argument]:
    return _get_metric_base_args()


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
