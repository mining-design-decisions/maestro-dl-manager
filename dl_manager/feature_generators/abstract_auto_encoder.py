import json
import os
import shutil
import abc

import issue_db_api


import keras.models
from .. import db_util
from .generator import AbstractFeatureGenerator, FeatureEncoding
from ..config.arguments import Argument, EnumArgument, IntArgument, QueryArgument
from ..config.constraints import Constraint
from ..model_io import InputEncoding
from .bow_frequency import BOWFrequency
from .bow_normalized import BOWNormalized
from .tfidf import TfidfGenerator
from ..logger import get_logger

log = get_logger('Abstract Auto Encoder')


class AbstractAutoEncoder(AbstractFeatureGenerator, abc.ABC):

    @staticmethod
    def input_encoding_type() -> InputEncoding:
        return InputEncoding.Vector

    @staticmethod
    @abc.abstractmethod
    def get_extra_params():
        pass

    @abc.abstractmethod
    def train_encoder(self, tokenized_issues: list[list[str]], metadata, args: dict[str, str]):
        pass

    def generate_vectors(self, tokenized_issues: list[list[str]], metadata, args: dict[str, str]):
        if self.pretrained is None:
            encoder = self.train_encoder(tokenized_issues, metadata, args)
        else:
            path = os.path.join(
                self.conf.get('predict.model'),
                self.conf.get('system.storage.auxiliary_prefix'),
                self.pretrained['encoder-model']
            )
            encoder = keras.models.load_model(path)
        ######################################################################
        # Plot Test Data
        log.info('Generating Testing Features')
        if self.pretrained is None:
            with open(self.conf.get('system.storage.generators')[-1]) as file:
                settings = json.load(file)
        else:
            a_map = self.conf.get('system.storage.auxiliary_map')
            with open(a_map[self.pretrained['wrapped-generator']]) as file:
                settings = json.load(file)
        features = self.prepare_features(keys=self.issue_keys,
                                         issues=tokenized_issues,
                                         settings=settings['settings'],
                                         generator_name=settings['generator'])[1]['features']
        log.info('Mapping testing features')
        as_2d = encoder.predict(features)
        # Debugging code: plotting features
        # log.info('Rendering plot')
        # import matplotlib.pyplot as pyplot
        # colors = numpy.asarray(self.colors)
        # pyplot.scatter(as_2d[:, 0][colors == 0], as_2d[:, 1][colors == 0], c='r', alpha=0.5, label='Executive')
        # pyplot.scatter(as_2d[:, 0][colors == 1], as_2d[:, 1][colors == 1], c='g', alpha=0.5, label='Property')
        # pyplot.scatter(as_2d[:, 0][colors == 2], as_2d[:, 1][colors == 2], c='b', alpha=0.5, label='Existence')
        # pyplot.scatter(as_2d[:, 0][colors == 3], as_2d[:, 1][colors == 3], c='y', alpha=0.5, label='Non-Architectural')
        # pyplot.legend(loc='upper left')
        # pyplot.show()
        # raise RuntimeError
        #import matplotlib.pyplot as pyplot
        #seaborn.heatmap(avg.reshape(37, 56), cmap='viridis')
        #pyplot.show()
        if self.pretrained is None:
            wrapped_generator = self.conf.get('system.storage.generators').pop(-1)
            encoder_dir = 'autoencoder'
            if os.path.exists(encoder_dir):
                shutil.rmtree(encoder_dir)
            os.makedirs(encoder_dir, exist_ok=True)
            encoder.save(encoder_dir)
            feature_size = self.params['target-feature-size']
            self.save_pretrained(
                {
                    'wrapped-generator': wrapped_generator,
                    'encoder-model': encoder_dir,
                    'feature-size': feature_size
                },
                [
                    os.path.join(path, f)
                    for path, _, files in os.walk(encoder_dir)
                    for f in files
                    if os.path.isfile(os.path.join(path, f))
                ] + [
                    wrapped_generator
                ]
            )
        else:
            feature_size = self.pretrained['feature-size']
        return {
            'features': as_2d.tolist(),
            'feature_shape': feature_size,
            'feature_encoding': {
                'encoding': self.feature_encoding(),
                'metadata': []
            }
        }

    def prepare_features(self, keys=None, issues=None, settings=None, generator_name=None):
        if True:
            if issues is None:
                query = db_util.json_to_query(self.params['training-data-query'])
                db: issue_db_api.IssueRepository = self.conf.get('system.storage.database-api')
                issues = [issue.summary + issue.description
                          for issue in db.search(query, attributes=['summary', 'description'])]
            if settings is None:
                params = self.params.copy()
                params['min-doc-count'] = params['bow-min-count']
                for name in self.get_extra_params():
                    try:
                        del params[name]
                    except KeyError:
                        pass
                match generator_name:
                    case 'BOWFrequency':
                        generator = BOWFrequency(self.conf, **params)
                    case 'BOWNormalized':
                        generator = BOWNormalized(self.conf, **params)
                    case 'TfidfGenerator':
                        try:
                            del params['min-doc-count']
                        except KeyError:
                            pass
                        generator = TfidfGenerator(self.conf, **params)
                    case _ as g:
                        raise ValueError(f'Unsupported feature generator for auto-encoder: {g}')
            else:
                match generator_name:
                    case 'BOWFrequency':
                        generator = BOWFrequency(self.conf, pretrained_generator_settings=settings)
                    case 'BOWNormalized':
                        generator = BOWNormalized(self.conf, pretrained_generator_settings=settings)
                    case 'TfidfGenerator':
                        generator = TfidfGenerator(self.conf, pretrained_generator_settings=settings)
                    case _ as g:
                        raise ValueError(f'Unsupported feature generator for auto-encoder: {g}')
        return keys, generator.generate_vectors(
            generator.preprocess(issues),
            [[] for _ in range(len(issues))],
            generator.params
        )
    @staticmethod
    def feature_encoding() -> FeatureEncoding:
        return FeatureEncoding.Numerical

    @classmethod
    def get_constraints(cls) -> list[Constraint]:
        return super().get_constraints()

    @classmethod
    def get_arguments(cls) -> dict[str, Argument]:
        return super(AbstractAutoEncoder, AbstractAutoEncoder).get_arguments() | {
            'training-data-query': QueryArgument(
                name='training-data-query',
                description='Query to retrieve data used to train the auto-encoder'
            ),
            'bow-min-count': IntArgument(
                name='bow-min-count',
                description='Minimum document count for bag of words',
                minimum=0,
                default=0
            ),
            'inner-generator': EnumArgument(
                name='inner-generator',
                description='Feature generator to transform issues to text',
                options=['BOWFrequency', 'BOWNormalized', 'TfidfGenerator']
            ),
        }
