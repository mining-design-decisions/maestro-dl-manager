##############################################################################
##############################################################################
# Imports
##############################################################################

from __future__ import annotations

import abc
import csv
import enum
import hashlib
import itertools
import json
import os.path
import random
import string
import json
import nltk
import psycopg2
from psycopg2 import sql
import issue_db_api

from ..config.arguments import (
    Argument,
    BoolArgument,
    IntArgument,
    StringArgument,
    EnumArgument,
    ArgumentConsumer,
)
from ..config.constraints import Constraint
from .util.text_cleaner import FormattingHandling, clean_issue_text, fix_contractions
from .. import accelerator
from ..model_io import InputEncoding, classification8_lookup
from ..custom_kfold import stratified_trim
from .util import ontology
from .util.technology_replacer import (
    replace_technologies,
    get_filename as get_technology_file_filename,
)
from ..config.core import Config
from ..logger import get_logger, timer
from ..data_manager import Dataset


log = get_logger("Base Feature Generator")

from ..data_manager_bootstrap import get_raw_text_file_name
import os

csv.field_size_limit(100000000)

POS_CONVERSION = {
    "JJ": "a",
    "JJR": "a",
    "JJS": "a",
    "NN": "n",
    "NNS": "n",
    "NNP": "n",
    "NNPS": "n",
    "RB": "r",
    "RBR": "r",
    "RBS": "r",
    "VB": "v",
    "VBD": "v",
    "VBG": "v",
    "VBN": "v",
    "VBP": "v",
    "VBZ": "v",
    "WRB": "r",
}

ATTRIBUTE_CONSTANTS = {
    "n_attachments": "n_attachments",
    "n_comments": "n_comments",
    "len_comments": "len_comments",
    "n_components": "n_components",
    "len_description": "len_description",
    "n_issuelinks": "n_issuelinks",
    "n_labels": "n_labels",
    "parent": "parent",
    "n_subtasks": "n_subtasks",
    "len_summary": "len_summary",
    "n_votes": "n_votes",
    "n_watches": "n_watches",
    "issuetype": "issuetype",
    "labels": "labels",
    "priority": "priority",
    "resolution": "resolution",
    "status": "status",
}

##############################################################################
##############################################################################
# Auxiliary Classes
##############################################################################


class FeatureEncoding(enum.Enum):
    Numerical = enum.auto()  # No metadata
    Categorical = enum.auto()  # No metadata
    Mixed = enum.auto()  # Metadata = Indices of categorical features
    Bert = enum.auto()  # No metadata

    def as_string(self):
        match self:
            case self.Numerical:
                return "numerical"
            case self.Categorical:
                return "categorical"
            case self.Mixed:
                return "mixed"
            case self.Bert:
                return "bert"

    def from_string(self, x: str):
        match x:
            case "numerical":
                return self.Numerical
            case "categorical":
                return self.Categorical
            case "mixed":
                return self.Mixed
            case "bert":
                return self.Bert
            case _:
                raise ValueError(f"Invalid feature encoding: {x}")


def _escape(x):
    for ws in string.whitespace:
        x = x.replace(ws, "_")
    x = x.replace(".", "dot")
    for illegal in "/<>:\"/\\|?*'":
        x = x.replace(illegal, "")
    return x


class _NullDict(dict):
    def __missing__(self, key):
        return key


##############################################################################
##############################################################################
# Main Class
##############################################################################


class AbstractFeatureGenerator(abc.ABC, ArgumentConsumer):
    def __init__(
        self,
        conf: Config,
        /,
        *,
        pretrained_generator_settings: dict | None = None,
        pretrained_must_include_labels=False,  # Might need to include labels when used to generate separate test set
        **params,
    ):
        self.__params = params
        self.__pretrained = pretrained_generator_settings
        self.__pretrained_with_labels = pretrained_must_include_labels
        self.__colors = None
        self.__keys = None
        self.conf = conf
        self.__ontology_classes = None
        self.__apply_ontologies = False
        self._project_names_file = ""
        self._name_lookup_file = ""
        if self.__pretrained is not None:
            if self.__params:
                raise ValueError(
                    "Feature generator does not take params when pretrained settings are given"
                )
            # Populate params for default pre-processing,
            # which does not require any trained settings.
            for name in AbstractFeatureGenerator.get_arguments():
                if name in self.__pretrained:
                    self.__params[name] = self.__pretrained[name]
            aux = self.conf.get("system.storage.auxiliary-map")
            if "ontology-classes" in self.__pretrained:
                # self.conf.set('run.ontology-classes', aux[self.__pretrained['ontology-classes']])
                self.__ontology_classes = aux[self.__pretrained["ontology-classes"]]
                self.__apply_ontologies = self.__pretrained["use-ontology-classes"]
            if x := self.__pretrained.get("$project-names-file", ""):
                self._project_names_file = aux[x]
            if x := self.__pretrained.get("$project-lookup-file", ""):
                self._name_lookup_file = aux[x]
        else:
            self.__ontology_classes = (
                f'{conf.get("system.storage.file-prefix")}_ontologies.json'
            )
            self.__have_ontology_classes = False
            if ident := conf.get("run.ontology-classes"):
                repo: issue_db_api.IssueRepository = conf.get(
                    "system.storage.database-api"
                )
                ontology_file = repo.get_file_by_id(ident)
                ontology_file.download(self.__ontology_classes)
                self.__apply_ontologies = conf.get("run.apply-ontology-classes")
                self.__have_ontology_classes = True

    def require_ontology_classes(self):
        if not self.__have_ontology_classes:
            raise ValueError("Need ontology classes")
        return self.__ontology_classes

    @property
    def params(self) -> dict[str, str]:
        return self.__params

    @property
    def pretrained(self) -> dict | None:
        return self.__pretrained

    @property
    def colors(self) -> list[int]:
        if self.__colors is None:
            raise RuntimeError("No colors yet")
        return self.__colors

    @property
    def issue_keys(self) -> list[str]:
        if self.__keys is None:
            raise RuntimeError("No keys yet")
        return self.__keys

    def save_pretrained(
        self, pretrained_settings: dict, auxiliary_files: list[str] = None
    ):
        if auxiliary_files is None:
            auxiliary_files = []
        log.info(f"Saving {self.__class__.__name__} feature encoding")
        settings = "_".join(f"{key}-{value}" for key, value in self.__params.items())
        filename = f"{self.__class__.__name__}__{settings}"
        prefix = self.conf.get("system.storage.file-prefix")
        filename = f"{prefix}_{hashlib.sha512(filename.encode()).hexdigest()}.json"
        filename = os.path.join(self.conf.get("system.os.scratch-directory"), filename)
        for name in AbstractFeatureGenerator.get_arguments():
            if name in self.__params:
                pretrained_settings[name] = self.__params[name]
        if self.__have_ontology_classes:
            pretrained_settings["ontology-classes"] = self.__ontology_classes
            self.conf.get("system.storage.auxiliary").append(self.__ontology_classes)
        pretrained_settings["use-ontology-classes"] = self.conf.get(
            "run.apply-ontology-classes"
        )
        if x := self.params["replace-other-technologies-list"]:
            pretrained_settings["$project-names-file"] = get_technology_file_filename(
                x, self.conf
            )
            self.conf.get("system.storage.auxiliary").append(
                get_technology_file_filename(x, self.conf)
            )
        if x := self.params["replace-this-technology-mapping"]:
            pretrained_settings["$project-lookup-file"] = get_technology_file_filename(
                x, self.conf
            )
            self.conf.get("system.storage.auxiliary").append(
                get_technology_file_filename(x, self.conf)
            )
        self.conf.get("system.storage.generators").append(filename)
        self.conf.get("system.storage.auxiliary").extend(auxiliary_files)
        with open(filename, "w") as file:
            json.dump(
                {
                    "settings": pretrained_settings,
                    "generator": self.__class__.__name__,
                },
                file,
            )

    @staticmethod
    @abc.abstractmethod
    def input_encoding_type() -> InputEncoding:
        """Type of input encoding generated by this generator."""

    @abc.abstractmethod
    def generate_vectors(
        self, tokenized_issues: list[list[str]], metadata, args: dict[str, str]
    ):
        # TODO: implement this method
        # TODO: this method should take in data, and generate
        # TODO: the corresponding feature vectors
        pass

    @staticmethod
    @abc.abstractmethod
    def feature_encoding() -> FeatureEncoding:
        pass

    @classmethod
    @abc.abstractmethod
    def get_constraints(cls) -> list[Constraint]:
        return []

    @classmethod
    @abc.abstractmethod
    def get_arguments(cls) -> dict[str, Argument]:
        return {
            "max-len": IntArgument(
                name="max-len",
                description="words limit of the issue text. Set to -1 to disable.",
                minimum=-1,
                default=-1,
            ),
            "disable-lowercase": BoolArgument(
                name="disable-lowercase",
                description="transform words to lowercase",
                default=False,
            ),
            "disable-stopwords": BoolArgument(
                name="disable-stopwords",
                description="remove stopwords from text",
                default=False,
            ),
            "use-stemming": BoolArgument(
                name="use-stemming",
                description="stem the words in the text",
                default=False,
            ),
            "use-lemmatization": BoolArgument(
                name="use-lemmatization",
                description="Use lemmatization on words in the text",
                default=False,
            ),
            "use-pos": BoolArgument(
                name="use-pos",
                description="Enhance words in the text with part of speech information",
                default=False,
            ),
            "class-limit": IntArgument(
                name="class-limit",
                description="limit the amount of items per class. Set to -1 to disable",
                default=-1,
                minimum=-1,
            ),
            "metadata-attributes": StringArgument(
                name="metadata-attributes",
                description="Comma-separated list of metadata attributes to fetch for use in feature generation",
                default="",
            ),
            "formatting-handling": EnumArgument(
                name="formatting-handling",
                description="How to handle formatting",
                options=["markers", "remove", "keep"],
                default="markers",
            ),
            "replace-this-technology-mapping": StringArgument(
                name="replace-this-technology-mapping",
                description="If given, should be a file mapping project keys to project names. "
                "Project names in text will be replacement with `this-technology-replacement`.",
                default="",
            ),
            "this-technology-replacement": StringArgument(
                name="this-technology-replacement",
                description="See description of `replace-this-technology-mapping`",
                default="",
            ),
            "replace-other-technologies-list": StringArgument(
                name="replace-other-technologies-list",
                description="If given, should be a file containing a list of project names. "
                "Project names will be replaced with `other-technology-replacement`",
                default="",
            ),
            "other-technology-replacement": StringArgument(
                name="other-technology-replacement",
                description="See description of `replace-other-technology-list`.",
                default="",
            ),
            "text-features-no-formatting-removal": BoolArgument(
                name="text-features-no-formatting-removal",
                description="If True, formatting is not removed for features of type `Text`.",
                default=False,
            ),
        }

    def load_data_from_db(self, query: issue_db_api.Query, metadata_attributes):
        # api: issue_db_api.IssueRepository = self.conf.get("system.storage.database-api")
        # issues = api.search(
        #     query,
        #     attributes=metadata_attributes + ["key", "summary", "description"],
        #     load_labels=True,
        # )
        
        
        # with open('dl_manager/feature_generators/data_json.json', 'r') as f:
        #     issues = json.load(f)
    
        # Create list to hold issue objects
        # issues = []
        
        # issues.sort(key=lambda i: i["id"])
        
        # PostgreSQL connection details (match the Docker Compose settings)
        pg_conn = psycopg2.connect(
            dbname="issues",
            user="postgres",
            password="pass",
            host="localhost",
            port="5432"
        )
        pg_cursor = pg_conn.cursor()
        
         
        # with open("dl_manager/feature_generators/sample_300.json", "r") as json_file:
        #     data = json.load(json_file)


        # Extract issue keys from the JSON data
        # issue_keys = {item.get('key') for item in data if item.get('key')}
        issue_keys = ['MAPREDUCE-4658', 'HDFS-11863', 'CASSANDRA-5498', 'YARN-10322', 'CASSANDRA-8409', 'CASSANDRA-5967', 'CASSANDRA-10091', 'HDFS-736', 'HDFS-5560', 'HDFS-1364', 'TAJO-2147', 'HDFS-10178', 'CASSANDRA-11569', 'CASSANDRA-6696', 'HDFS-15039', 'HADOOP-15218', 'HDFS-14183', 'CASSANDRA-14667', 'HADOOP-7305', 'YARN-1637', 'HADOOP-5636', 'YARN-1001', 'CASSANDRA-199', 'YARN-10466', 'HDFS-5838', 'TAJO-280', 'HADOOP-15595', 'CASSANDRA-16030', 'HDFS-5108', 'MAPREDUCE-2053', 'CASSANDRA-14799', 'HDFS-9802', 'YARN-1149', 'CASSANDRA-7907', 'CASSANDRA-3801', 'HADOOP-6915', 'CASSANDRA-5386', 'CASSANDRA-1722', 'HADOOP-5932', 'CASSANDRA-3578', 'TAJO-1788', 'HADOOP-9874', 'CASSANDRA-18060', 'YARN-953', 'HADOOP-15994', 'HADOOP-59', 'HADOOP-12945', 'MAPREDUCE-5286', 'TAJO-169', 'CASSANDRA-1440', 'HADOOP-13385', 'HDFS-6091', 'YARN-168', 'TAJO-1281', 'HADOOP-15285', 'HDFS-8137', 'HADOOP-291', 'YARN-11205', 'CASSANDRA-17920', 'HADOOP-18302', 'YARN-8134', 'CASSANDRA-2450', 'HADOOP-13912', 'HDFS-2942', 'CASSANDRA-16640', 'CASSANDRA-9662', 'HDFS-6492', 'CASSANDRA-6473', 'CASSANDRA-10065', 'MAPREDUCE-2403', 'YARN-1744', 'CASSANDRA-3792', 'HDFS-6779', 'CASSANDRA-16488', 'HADOOP-9987', 'HADOOP-10676', 'MAPREDUCE-5177', 'HDFS-14370', 'HDFS-7859', 'HADOOP-14124', 'HADOOP-10776', 'CASSANDRA-1576', 'MAPREDUCE-5496', 'YARN-4313', 'CASSANDRA-2181', 'HDFS-1898', 'HDFS-6182', 'HDFS-5542', 'TAJO-176', 'HADOOP-15628', 'HDFS-9550', 'HADOOP-12994', 'CASSANDRA-2677', 'HDFS-10712', 'HDFS-3053', 'CASSANDRA-5631', 'MAPREDUCE-1740', 'HDFS-6482', 'HDFS-12461', 'CASSANDRA-1408', 'HADOOP-17220', 'HDFS-15425', 'HADOOP-10601', 'TAJO-1209', 'YARN-3115', 'CASSANDRA-8319', 'YARN-11266', 'CASSANDRA-12032', 'CASSANDRA-8521', 'YARN-8700', 'HDFS-9740', 'MAPREDUCE-6397', 'HADOOP-15015', 'HADOOP-17975', 'TAJO-1087', 'CASSANDRA-17265', 'TAJO-307', 'YARN-4999', 'CASSANDRA-10305', 'HADOOP-2220', 'HDFS-12128', 'HDFS-15608', 'YARN-10028', 'HADOOP-5303', 'CASSANDRA-1969', 'CASSANDRA-3968', 'CASSANDRA-11704', 'YARN-268', 'CASSANDRA-16456', 'CASSANDRA-13570', 'HDFS-7888', 'TAJO-1019', 'CASSANDRA-17600', 'CASSANDRA-8494', 'CASSANDRA-13122', 'HDFS-12380', 'CASSANDRA-2817', 'HDFS-8968', 'HDFS-5113', 'CASSANDRA-12299', 'HDFS-4989', 'CASSANDRA-12643', 'CASSANDRA-2331', 'HADOOP-5227', 'YARN-4791', 'YARN-2878', 'CASSANDRA-16249', 'CASSANDRA-12700', 'CASSANDRA-15922', 'CASSANDRA-3887', 'HDFS-5366', 'CASSANDRA-16577', 'CASSANDRA-6144', 'HDFS-9507', 'HADOOP-13518', 'HADOOP-12207', 'YARN-692', 'HADOOP-17051', 'HDFS-545', 'YARN-3039', 'YARN-5763', 'MAPREDUCE-430', 'HADOOP-10963', 'CASSANDRA-7770', 'HADOOP-350', 'HADOOP-5401', 'HDFS-1543', 'HADOOP-6493', 'CASSANDRA-15584', 'MAPREDUCE-1413', 'YARN-9679', 'CASSANDRA-17419', 'YARN-2215', 'CASSANDRA-8525', 'HDFS-6526', 'YARN-4683', 'YARN-382', 'HADOOP-6195', 'MAPREDUCE-267', 'CASSANDRA-10653', 'HADOOP-1701', 'YARN-10383', 'TAJO-1728', 'YARN-7020', 'HADOOP-10764', 'HADOOP-5896', 'CASSANDRA-1373', 'HADOOP-6676', 'CASSANDRA-6886', 'YARN-1075', 'CASSANDRA-1470', 'MAPREDUCE-1407', 'CASSANDRA-607', 'HDFS-386', 'HADOOP-10655', 'HDFS-16613', 'YARN-10436', 'HADOOP-14667', 'HDFS-107', 'CASSANDRA-7813', 'HADOOP-17819', 'CASSANDRA-4600', 'YARN-3003', 'CASSANDRA-14464', 'CASSANDRA-12567', 'YARN-4125', 'TAJO-521', 'CASSANDRA-2865', 'CASSANDRA-336', 'YARN-1815', 'TAJO-1625', 'CASSANDRA-18005', 'CASSANDRA-8431', 'YARN-1799', 'CASSANDRA-18176', 'HDFS-8323', 'HADOOP-234', 'HADOOP-14261', 'CASSANDRA-746', 'HDFS-1445', 'MAPREDUCE-6376', 'CASSANDRA-10154', 'CASSANDRA-13045', 'YARN-3611', 'YARN-2138', 'HDFS-4146', 'MAPREDUCE-6962', 'TAJO-1527', 'CASSANDRA-343', 'HADOOP-3991', 'TAJO-1844', 'HADOOP-5732', 'CASSANDRA-8030', 'TAJO-1883', 'HDFS-5656', 'YARN-11252', 'CASSANDRA-12526', 'MAPREDUCE-4851', 'YARN-9433', 'HADOOP-2768', 'HADOOP-10693', 'CASSANDRA-3723', 'YARN-7773', 'HADOOP-9194', 'CASSANDRA-5163', 'HDFS-5390', 'HDFS-4677', 'CASSANDRA-18090', 'HDFS-2035', 'TAJO-1062', 'HADOOP-7235', 'HADOOP-11169', 'TAJO-1701', 'HADOOP-14188', 'MAPREDUCE-2201', 'TAJO-2034', 'CASSANDRA-4408', 'HADOOP-9365', 'CASSANDRA-11978', 'CASSANDRA-1035', 'CASSANDRA-8574', 'CASSANDRA-9974', 'TAJO-889', 'CASSANDRA-4981', 'MAPREDUCE-500', 'HADOOP-17624', 'CASSANDRA-12282', 'HDFS-13163', 'HADOOP-14584', 'HADOOP-7224', 'HDFS-3152', 'HDFS-14462', 'TAJO-1861', 'CASSANDRA-1308', 'CASSANDRA-4383', 'CASSANDRA-11853', 'CASSANDRA-10213', 'HADOOP-4461', 'MAPREDUCE-3687', 'HADOOP-10039', 'CASSANDRA-5777', 'CASSANDRA-17061', 'HDFS-4658', 'CASSANDRA-17398', 'HADOOP-14005', 'HADOOP-11975', 'YARN-7263', 'HDFS-12016', 'CASSANDRA-11613', 'HADOOP-12623', 'CASSANDRA-2792', 'HDFS-2187', 'TAJO-2181', 'YARN-2684', 'HADOOP-6848', 'HDFS-9256', 'HDFS-1274', 'HADOOP-8345', 'HDFS-9638', 'HADOOP-14764', 'HDFS-8671', 'TAJO-99', 'YARN-9008', 'YARN-417', 'HADOOP-16253', 'HADOOP-12176', 'HADOOP-11540', 'TAJO-1385', 'TAJO-2159', 'HADOOP-8130', 'CASSANDRA-12707', 'TAJO-1807', 'HADOOP-4108', 'HDFS-2064', 'TAJO-2137', 'HADOOP-7924', 'TAJO-856', 'YARN-7192', 'CASSANDRA-6422', 'CASSANDRA-11748', 'HADOOP-17775', 'YARN-1440', 'YARN-2173', 'CASSANDRA-5234', 'HDFS-16355', 'YARN-4129', 'HADOOP-8683', 'HDFS-3440', 'CASSANDRA-10038', 'YARN-6058', 'TAJO-810', 'CASSANDRA-17753', 'HDFS-15632', 'CASSANDRA-4913', 'CASSANDRA-14701', 'MAPREDUCE-1679', 'CASSANDRA-15399', 'YARN-11277', 'MAPREDUCE-352', 'HDFS-11403', 'HDFS-13804', 'HADOOP-8826', 'YARN-5', 'YARN-7117', 'HDFS-1720', 'YARN-7354', 'MAPREDUCE-4854', 'CASSANDRA-7024', 'HADOOP-10067', 'CASSANDRA-1643', 'HADOOP-10099', 'CASSANDRA-17148', 'TAJO-1425', 'CASSANDRA-13530', 'YARN-5221', 'CASSANDRA-5016', 'CASSANDRA-7897', 'CASSANDRA-12668', 'CASSANDRA-9894', 'CASSANDRA-1657', 'CASSANDRA-70', 'YARN-9981', 'HADOOP-13842', 'HDFS-16317', 'CASSANDRA-5150', 'CASSANDRA-17031', 'CASSANDRA-14185', 'HDFS-16697', 'CASSANDRA-3762', 'TAJO-1252', 'MAPREDUCE-228', 'CASSANDRA-6448', 'HADOOP-16816', 'HADOOP-18498', 'CASSANDRA-5202', 'MAPREDUCE-1947', 'CASSANDRA-8538', 'CASSANDRA-6709', 'MAPREDUCE-1072', 'CASSANDRA-2043', 'CASSANDRA-13316', 'CASSANDRA-16059', 'YARN-2800', 'HDFS-6836', 'TAJO-1148', 'HDFS-7499', 'HADOOP-11501', 'HADOOP-3177', 'TAJO-1784', 'HADOOP-10708', 'CASSANDRA-12343', 'MAPREDUCE-3347', 'HDFS-9607', 'HADOOP-10660', 'CASSANDRA-1084', 'HADOOP-16878', 'CASSANDRA-10490', 'YARN-9816', 'MAPREDUCE-1939', 'YARN-2814', 'YARN-5752', 'HDFS-8531', 'CASSANDRA-1946', 'CASSANDRA-8826']
        # Prepare the SQL query with placeholders for the issue_ids
        query = """
        SELECT id, issue_id, body 
        FROM issues_comments 
        WHERE is_bot = false 
        AND LENGTH(body) > 200 
        AND issue_id = ANY(%s)
        ORDER BY id;
        """

        # Execute the query with the issue_keys as parameter
        pg_cursor.execute(query, (list(issue_keys),))
        comments = pg_cursor.fetchall()
        if not comments:
            return
        
        print(comments[0][1], comments[0][0])

        labels = {
            "detection": [],
            "classification3": [],
            "classification3simplified": [],
            "classification8": [],
            "issue_keys": [comment[1] for comment in comments],
            "issue_ids": [comment[0] for comment in comments],
        }
        classification_indices = {
            "Existence": [],
            "Property": [],
            "Executive": [],
            "Non-Architectural": [],
        }
        if self.pretrained is None or self.__pretrained_with_labels:
            raw_labels = [issue.manual_label for issue in comments]
            for index, raw in enumerate(raw_labels):
                self.update_labels(
                    labels,
                    classification_indices,
                    index,
                    raw.existence,
                    raw.executive,
                    raw.property,
                )
        texts = []
        for issue in comments:
            # summary = x if (x := issue.pop('summary')) is not None else ''
            # description = x if (x := issue.pop('description')) is not None else ''
            # labels['issue_keys'].append(issue.pop('key'))
            texts.append([issue[2]])
        metadata = []
        for issue in comments:
            metadata.append(
                {attr: getattr(issue, attr) for attr in metadata_attributes}
            )
        return texts, metadata, labels, classification_indices

    def update_labels(
        self,
        labels,
        classification_indices,
        current_index,
        is_existence,
        is_executive,
        is_property,
    ):
        if self.__colors is None:
            self.__colors = []
        if is_executive:  # Executive
            labels["classification3simplified"].append((0, 1, 0, 0))
            classification_indices["Executive"].append(current_index)
            self.__colors.append(0)
        elif is_property:  # Property
            labels["classification3simplified"].append((0, 0, 1, 0))
            classification_indices["Property"].append(current_index)
            self.__colors.append(1)
        elif is_existence:  # Existence
            labels["classification3simplified"].append((1, 0, 0, 0))
            classification_indices["Existence"].append(current_index)
            self.__colors.append(2)
        else:  # Non-architectural
            labels["classification3simplified"].append((0, 0, 0, 1))
            classification_indices["Non-Architectural"].append(current_index)
            self.__colors.append(3)

        if is_executive or is_property or is_existence:
            labels["detection"].append(True)
        else:
            labels["detection"].append(False)

        key = (is_existence, is_executive, is_property)
        labels["classification8"].append(classification8_lookup[key])
        labels["classification3"].append(key)

    def generate_features(self, query: issue_db_api.Query, output_mode: str):
        """Generate features from the data in the given source file,
        and store the results in the given target file.
        """
        metadata_attributes = [
            attr for attr in self.__params["metadata-attributes"].split(",") if attr
        ]
        for attr in metadata_attributes:
            if attr not in ATTRIBUTE_CONSTANTS:
                raise ValueError(f"Unknown metadata attribute: {attr}")

        texts, metadata, labels, classification_indices = self.load_data_from_db(
            query, metadata_attributes
        )

        limit = self.params["class-limit"]
        if limit != -1 and self.pretrained is None:  # Only execute if not pretrained
            stratified_indices = []
            for issue_type in classification_indices.keys():
                project_labels = [
                    label
                    for index, label in enumerate(
                        [label.split("-")[0] for label in labels["issue_keys"]]
                    )
                    if index in classification_indices[issue_type]
                ]
                trimmed_indices = stratified_trim(limit, project_labels)
                stratified_indices.extend(
                    [classification_indices[issue_type][idx] for idx in trimmed_indices]
                )
            texts = [
                text for idx, text in enumerate(texts) if idx in stratified_indices
            ]
            for key in labels.keys():
                labels[key] = [
                    label
                    for idx, label in enumerate(labels[key])
                    if idx in stratified_indices
                ]

        # The replace_technologies function
        # already performs the substitutions conditionally,
        # so it can be in the main code path here.
        if self.pretrained is None:
            texts = replace_technologies(
                issues=texts,
                keys=labels["issue_keys"],
                project_names_ident=self.params["replace-other-technologies-list"],
                project_name_lookup_ident=self.params[
                    "replace-this-technology-mapping"
                ],
                this_project_replacement=self.params["this-technology-replacement"],
                other_project_replacement=self.params["other-technology-replacement"],
                conf=self.conf,
            )
        else:
            # Cry
            texts = replace_technologies(
                issues=texts,
                keys=labels["issue_keys"],
                project_names_ident=None,
                project_name_lookup_ident=None,
                this_project_replacement=self.params.get(
                    "this-technology-replacement", ""
                ),
                other_project_replacement=self.params.get(
                    "other-technology-replacement", ""
                ),
                conf=self.conf,
                project_names_file=self._project_names_file,
                name_lookup_file=self._name_lookup_file,
            )

        if False and self.input_encoding_type() == InputEncoding.Text:
            # summaries, descriptions = (list(x) for x in zip(*texts))
            # texts = (list(x) for x in zip(*texts))
            print("without preprocessing")
            texts = [text[0] for text in texts]
            if not self.__params.get("text-features-no-formatting-removal",False):
                handling_string = self.__params["formatting-handling"]
                handling = FormattingHandling.from_string(handling_string)
                texts, descriptions = self.remove_formatting(
                    texts, [], handling
                )
            # summaries, descriptions = (list(x) for x in zip(*texts))
            tokenized_issues = [
                [f"{text}"]
                for text in texts
            ]
        else:
            # with cProfile.Profile() as p:
            #    tokenized_issues = self.preprocess(texts)
            # p.dump_stats('profile.txt')
            tokenized_issues = self.preprocess(texts, labels["issue_keys"])

        log.info("Generating feature vectors")
        with timer("Feature Generation"):
            output = self.generate_vectors(tokenized_issues, metadata, self.__params)
        output["labels"] = labels  # labels is empty when pretrained

        output["original"] = tokenized_issues
        if (
            "original" in output and not self.pretrained
        ):  # Only dump original text when not pre-trained.
            with open(get_raw_text_file_name(self.conf), "w") as file:
                mapping = {
                    key: text
                    for key, text in zip(labels["issue_ids"], output["original"])
                }
                json.dump(mapping, file)
            del output["original"]
        elif "original" in output:
            del output["original"]

        return Dataset(
            features=output["features"],
            labels=output["labels"][output_mode.lower()],
            shape=output["feature_shape"],
            embedding_weights=output.get("weights", None),
            vocab_size=output.get("vocab_size", None),
            weight_vector_length=output.get("word_vector_length", None),
            binary_labels=output["labels"]["detection"],
            issue_keys=output["labels"]["issue_keys"],
            ids=output["labels"]["issue_ids"],
        )

    def remove_formatting(
        self,
        summaries: list[str],
        descriptions: list[str],
        handling: FormattingHandling,
    ) -> tuple[list[str], list[str]]:
        summaries = accelerator.bulk_clean_text_parallel(
            summaries,
            handling.as_string(),
            self.conf.get("system.resources.threads"),
        )
        summaries = [fix_contractions(summary) for summary in summaries]
        descriptions = accelerator.bulk_clean_text_parallel(
            descriptions,
            handling.as_string(),
            self.conf.get("system.resources.threads"),
        )
        descriptions = [fix_contractions(description) for description in descriptions]
        return summaries, descriptions

    def preprocess(self, issues, issue_keys):
        log.info("Preprocessing Features")
        with timer("Feature Preprocessing"):
            if self.__apply_ontologies:
                ontology_table = ontology.load_ontology(self.__ontology_classes)
            else:
                ontology_table = None

            stopwords = nltk.corpus.stopwords.words("english")
            use_stemming = self.__params["use-stemming"]
            use_lemmatization = self.__params["use-lemmatization"]
            use_pos = self.__params["use-pos"]
            stemmer = nltk.stem.PorterStemmer()
            lemmatizer = nltk.stem.WordNetLemmatizer()
            use_lowercase = not self.__params["disable-lowercase"]
            use_ontologies = self.__apply_ontologies
            handling_string = self.__params["formatting-handling"]
            handling = FormattingHandling.from_string(handling_string)
            weights, tagdict, classes = nltk.load(
                "taggers/averaged_perceptron_tagger/averaged_perceptron_tagger.pickle"
            )
            tagger = accelerator.Tagger(weights, classes, tagdict)

            # summaries = (list(x) for x in zip(*issues))
            summaries = [text[0] for text in issues]
            summaries, descriptions = self.remove_formatting(
                summaries, [], handling
            )
            summaries = [clean_issue_text(summary) for summary in summaries]
            # descriptions = [
            #     clean_issue_text(description) for description in descriptions
            # ]
            texts = [
                [nltk.word_tokenize(sent.lower() if use_lowercase else sent) for sent in summary]
                for summary in summaries
            ]
            # old version which operated on tokens. No clue why I did that
            # texts = replace_technologies(
            #     keys=issue_keys,
            #     issues=texts,
            #     project_names_ident=self.params['replace-other-technologies-list'],
            #     project_name_lookup_ident=self.params['replace-this-technology-mapping'],
            #     this_project_replacement=self.params['this-technology-replacement'].split(),
            #     other_project_replacement=self.params['other-technology-replacement'].split(),
            #     conf=self.conf
            # )
            tagged = tagger.bulk_tag_parallel(
                texts, self.conf.get("system.resources.threads")
            )
            tokenized_issues = []
            for issue in tagged:
                all_words = []

                # Tokenize
                for words in issue:
                    # Apply ontology simplification. Must be done before stemming/lemmatization
                    if use_ontologies:
                        # assert ontology_table is not None, 'Missing --ontology-classes'
                        words = ontology.apply_ontologies_to_sentence(
                            words, ontology_table
                        )

                    # Remove stopwords
                    if not self.__params["disable-stopwords"]:
                        words = [
                            (word, tag) for word, tag in words if word not in stopwords
                        ]

                    if use_stemming and use_lemmatization:
                        raise ValueError("Cannot use both stemming and lemmatization")

                    if use_stemming:
                        words = [(stemmer.stem(word), tag) for word, tag in words]

                    if use_lemmatization:
                        words = [
                            (
                                lemmatizer.lemmatize(
                                    word, pos=POS_CONVERSION.get(tag, "n")
                                ),
                                tag,
                            )
                            for word, tag in words
                        ]

                    if use_pos:
                        words = [
                            f"{word}_{POS_CONVERSION.get(tag, tag)}"
                            for word, tag in words
                        ]
                    else:
                        words = [word for word, _ in words]

                    # At this point, we forget about sentence order
                    all_words.extend(words)

                # Limit issue length
                if (m := self.__params["max-len"]) > 0:
                    if len(all_words) > m:
                        all_words = all_words[0:m]

                tokenized_issues.append(all_words)

        log.info("Finished preprocessing")
        return tokenized_issues
