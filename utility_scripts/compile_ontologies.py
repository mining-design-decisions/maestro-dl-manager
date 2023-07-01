#!/usr/bin/python3

import csv
import json
import os.path

import nltk

SOURCE_DIR = '../data/ontologies'
TARGET_FILE = '../dl_manager/feature_generators/util/ontologies.json'


def txt_to_ontology(filename):
    with open(filename) as file:
        words = [line.strip().lower() for line in file if line.strip()]
        words = list(set([nltk.stem.WordNetLemmatizer().lemmatize(word) for word in words]))
    return {
        'name': get_class_name_from_filename(filename),
        'content': words
    }


def csv_to_ontology(filename):
    return txt_to_ontology(filename)    # all csv files are simple lists


def get_class_name_from_filename(path: str) -> str:
    _, filename = os.path.split(path)
    class_name, _ = os.path.splitext(filename)
    return class_name.replace(' ', '').replace('-', '').replace('_', '').lower()


def main():
    ontology_classes = []
    for filename in os.listdir(SOURCE_DIR):
        path = os.path.join(SOURCE_DIR, filename)
        if path.endswith('.csv'):
            ontology_classes.append(csv_to_ontology(path))
        elif path.endswith('.txt'):
            ontology_classes.append(txt_to_ontology(path))
        else:
            raise ValueError(f'Unknown extension: {path}')
    with open(TARGET_FILE, 'w') as file:
        json.dump(ontology_classes, file, indent=2)


main()