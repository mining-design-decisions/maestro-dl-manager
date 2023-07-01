import collections
import functools
import itertools
import random
import warnings

import numpy
import sklearn.model_selection


def stratified_split(data: numpy.ndarray, labels, split_size: float):
    splitter = sklearn.model_selection.StratifiedShuffleSplit(n_splits=1,
                                                              test_size=split_size)
    train, test = next(splitter.split(data, labels))
    return data[train], data[test]


def stratified_kfold(k: int, labels: list) -> list[list[int]]:
    # Generate 1 additional split for the test set
    folds = _get_stratified_splits(k, labels)
    indices = frozenset(range(0, k))
    for validation_set_index in range(0, k):
        test_set_index = (validation_set_index + 1) % k
        training_set_indices = indices - {validation_set_index, test_set_index}
        validation_set = folds[validation_set_index]
        test_set = folds[test_set_index]
        training_set = functools.reduce(
            list.__add__,
            (folds[index] for index in training_set_indices),
            []
        )
        yield training_set, validation_set, test_set


def stratified_kfold2(k: int, labels: list) -> list[list[int]]:
    folds = _get_stratified_splits(k, labels)
    indices = frozenset(range(0, k))
    for test_set_index in range(0, k):
        training_set_indices = indices - {test_set_index}
        test_set = folds[test_set_index]
        training_set = functools.reduce(
            list.__add__,
            (folds[index] for index in training_set_indices),
            []
        )
        yield training_set, test_set


def _get_stratified_splits(k: int, labels: list) -> list[list[int]]:
    # Compute Label Indices
    indices_per_label = collections.defaultdict(list)
    for index, label in enumerate(labels):
        indices_per_label[label].append(index)
    # Now, compute the optimal amount of items from each
    # label per fold, where any "remaining" items are
    # distributed in a round-robin fashion.
    label_amounts_per_fold = [
        {
            label: len(indices) // k
            for label, indices in indices_per_label.items()
        }
        for _ in range(k)
    ]
    # Now, distribute the remaining labels in
    # round-robin fashion.
    fold_index = 0
    for label, indices in indices_per_label.items():
        remaining = len(indices) % k
        for _ in range(remaining):
            label_amounts_per_fold[fold_index][label] += 1
            fold_index = (fold_index + 1) % k
    # Now, construct the actual folds by sampling from the
    # labels. We do not actually sample, but shuffle the
    # indices first and then popping from the lists.
    for indices in indices_per_label.values():
        random.shuffle(indices)
    folds = [_sample_from_labels(indices_per_label, label_amounts)
             for label_amounts in label_amounts_per_fold]
    # Shuffle all the folds
    for fold in folds:
        random.shuffle(fold)
    # Return result
    return folds


def _sample_from_labels(indices_per_label: dict,
                        label_amounts: dict) -> list:
    fold = []
    for label, amount in label_amounts.items():
        samples = [indices_per_label[label].pop() for _ in range(amount)]
        fold.extend(samples)
    return fold


def stratified_trim(max_size, labels, *, shuffle=True):
    # Early exit if we do not have enough samples
    if len(labels) <= max_size:
        warnings.warn(f'# Samples <= max_size: {len(labels)} <= {max_size}')
        return list(range(len(labels)))
    # Compute indices per label
    indices_per_label = collections.defaultdict(list)
    for index, label in enumerate(labels):
        indices_per_label[label].append(index)
    # Compute desired frequencies
    frequencies_per_label = {
        label: len(indices) / len(labels)
        for label, indices in indices_per_label.items()
    }
    if shuffle:
        for indices in indices_per_label.values():
            random.shuffle(indices)
    # Add as many samples as possible, making sure that we
    # do not overshoot the amount of samples added per class.
    new_indices_per_class = collections.defaultdict(list)
    for label, target_frequency in frequencies_per_label.items():
        while (len(new_indices_per_class[label]) + 1) / max_size <= target_frequency:
            new_indices_per_class[label].append(indices_per_label[label].pop())
    # Now, add all remaining samples. We add samples from the
    # class with the largest difference between current and
    # goal frequency
    classes = list(indices_per_label)
    while True:
        remaining = max_size - sum(
            len(indices) for indices in new_indices_per_class.values()
        )
        if remaining == 0:
            break
        to_add = max(
            classes,
            key=lambda c: frequencies_per_label[c] - len(new_indices_per_class[c])/max_size
        )
        new_indices_per_class[to_add].append(indices_per_label[to_add].pop())
    # Now, assemble all indices
    new_indices = functools.reduce(
        list.__add__,
        (list(indices) for indices in new_indices_per_class.values()),
        []
    )
    if shuffle:
        random.shuffle(new_indices)
    return new_indices


def round_robin_trim(limit, labels):
    if len(labels) < limit:
        return [idx for idx in range(0, len(labels))]

    projects = dict()
    # Gather indices per project
    for idx, label in enumerate(labels):
        if label not in projects:
            projects[label] = []
        projects[label].append(idx)

    # Randomly order the lists
    for key in projects.keys():
        random.shuffle(projects[key])

    generator = itertools.cycle(projects.keys())
    indices = []
    while len(indices) < limit:
        project = next(generator)
        if len(projects[project]) == 0:
            continue
        indices.append(projects[project].pop())

    return indices
