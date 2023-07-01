Command - `run` 

---

The `run` command is the central command in the deep learning manager.
It can be used to train classifiers and evaluate their performance.
Additionally, this command can also generate feature vectors -- 
avoiding the need to also call `make-features`.

We start out with the help message, which can be displayed by running
`python __main__.py run -h`:

``` 
usage: __main__.py run [-h] [--epochs EPOCHS] [--split-size SPLIT_SIZE] [--max-train MAX_TRAIN] [--k-cross K_CROSS] [--quick-cross]
                       [--cross-is-cross-project] [--force-regenerate-data] [--architectural-only]
                       [--hyper-params HYPER_PARAMS [HYPER_PARAMS ...]] [--test-project TEST_PROJECT] [--test-study TEST_STUDY]
                       [--class-balancer CLASS_BALANCER] [--batch-size BATCH_SIZE] [--peregrine]
                       [--combination-strategy {add,subtract,min,max,multiply,dot,concat}]
                       [--ensemble-strategy {stacking,boosting,voting}] [--stacking-meta-classifier STACKING_META_CLASSIFIER]
                       [--stacking-meta-classifier-hyper-parameters STACKING_META_CLASSIFIER_HYPER_PARAMETERS [STACKING_META_CLASSIFIER_HYPER_PARAMETERS ...]]
                       [--stacking-use-concat] [--stacking-no-matrix] [--boosting-rounds BOOSTING_ROUNDS] [--use-early-stopping]
                       [--early-stopping-patience EARLY_STOPPING_PATIENCE]
                       [--early-stopping-min-delta EARLY_STOPPING_MIN_DELTA [EARLY_STOPPING_MIN_DELTA ...]]
                       [--early-stopping-attribute EARLY_STOPPING_ATTRIBUTE [EARLY_STOPPING_ATTRIBUTE ...]] [--test-separately]
                       [--input-mode INPUT_MODE [INPUT_MODE ...]] [--output-mode OUTPUT_MODE] [--file FILE]
                       [--params PARAMS [PARAMS ...]] [--ontology-classes ONTOLOGY_CLASSES] [--apply-ontology-classes]
                       classifier [classifier ...]

positional arguments:
  classifier            Classifier to use. Use `list` for options

options:
  -h, --help            show this help message and exit
  --epochs EPOCHS, -e EPOCHS
                        Amount of training epochs
  --split-size SPLIT_SIZE, -s SPLIT_SIZE
                        Size of testing and validation splits.
  --max-train MAX_TRAIN
                        Maximum amount of training items. -1 for infinite
  --k-cross K_CROSS, -k K_CROSS
                        Enable k-fold cross-validation.
  --quick-cross, -qc    Enable k-fold cross validation
  --cross-is-cross-project
                        k-cross should be cross-project validation.
  --force-regenerate-data, -fr
                        Force regeneration of data.
  --architectural-only, -ao
                        If specified, only architectural issues are used
  --hyper-params HYPER_PARAMS [HYPER_PARAMS ...], -hp HYPER_PARAMS [HYPER_PARAMS ...]
                        Hyper-parameters params. Items in the name=value format.
  --test-project TEST_PROJECT
                        Name of project to be used as the test set
  --test-study TEST_STUDY
                        Name of the study to be used as the test set
  --class-balancer CLASS_BALANCER
                        Enable Class-Balancing
  --batch-size BATCH_SIZE
                        Specify the batch size used during training
  --peregrine           Specify to enable running on peregrine
  --combination-strategy {add,subtract,min,max,multiply,dot,concat}, -cs {add,subtract,min,max,multiply,dot,concat}
                        Strategy used to combine models. Use `combination-strategies` for more information.
  --ensemble-strategy {stacking,boosting,voting}, -es {stacking,boosting,voting}
                        Strategy used to combine models. Use `combination-strategies` for more information.
  --stacking-meta-classifier STACKING_META_CLASSIFIER
                        Classifier to use as meta-classifier in stacking.
  --stacking-meta-classifier-hyper-parameters STACKING_META_CLASSIFIER_HYPER_PARAMETERS [STACKING_META_CLASSIFIER_HYPER_PARAMETERS ...]
                        Hyper-parameters for the meta-classifier
  --stacking-use-concat
                        Use simple concatenation to create the input for the meta classifier
  --stacking-no-matrix  Disallow the use of matrices for meta classifier input
  --boosting-rounds BOOSTING_ROUNDS
                        Amount of rounds in the boosting process
  --use-early-stopping  If specified, use early stopping.
  --early-stopping-patience EARLY_STOPPING_PATIENCE
                        Patience used when using early stopping
  --early-stopping-min-delta EARLY_STOPPING_MIN_DELTA [EARLY_STOPPING_MIN_DELTA ...]
                        Minimum delta used when using early stopping. One entry for every attribute used.
  --early-stopping-attribute EARLY_STOPPING_ATTRIBUTE [EARLY_STOPPING_ATTRIBUTE ...]
                        Attribute(s) to use for early stopping (from the validation set)
  --test-separately     If given, disable combining multiple classifiers. In stead, test them separately on the same data.
  --input-mode INPUT_MODE [INPUT_MODE ...], -i INPUT_MODE [INPUT_MODE ...]
                        Generator to use. Use `list` for options.
  --output-mode OUTPUT_MODE, -o OUTPUT_MODE
                        Output mode to use. Use `list` for options.
  --file FILE, -f FILE  Data input file.
  --params PARAMS [PARAMS ...], -p PARAMS [PARAMS ...]
                        Generator params. Items in the name=value format.
  --ontology-classes ONTOLOGY_CLASSES
                        Path to a file containing ontology classes.
  --apply-ontology-classes
                        Enable application of ontology classes
```

In the remainder of this section, we will cover and explain all these 
arguments in more detail.

---

# Feature Generation Arguments 

The arguments `--file`, `--input-mode`, `--output-mode`,
`--params`, `--ontology-classes`, and 
`--apply-ontology-classes` are the same as those for the 
[make-features](./make-features.md) command. See the 
[documentation of the make-features command](./make-features.md) for 
more details about them.

Additionally, there are the `--force-regenerate-data` and 
`--architectural-only` arguments. 

If `--force-regenerate-data` is not 
given, the program will first check if is has generated the same 
feature vectors before and only generate them if it has not. If 
`--force-regenerate-data` is given, it will always generate new 
feature vectors.

`--architectural-only` can be used to only include architectural 
issues in the training process. The intended use case is to train
for the  classification task without the presence 
of non-architectural issues.

# Model Building Arguments

The positional argument `classifier`, and the arguments 
`--hyper-params`, `--combination-strategy`, `--ensemble-strategy`,
`--stacking-meta-classifier`, `--stacking-meta-classifier-hyper-parameters`,
`--stacking-use-concat`, `--stacking-no-matrix`, `--test-separately`,
and `--boosting-rounds` all deal with the construction 
and setup of the classifier.

The `classifier` argument must be supplied with the names of one or 
multiple classifiers. A list of classifiers can be obtained using the 
[list](./list.md) command. For every classifier, a separate feature 
vector generate must be given to `--input-mode`.

The `--hyper-params` argument is used to specify the hyperparameters 
of the classifiers. This argument must be specified in the following form:

``` 
--hyper-params param1=value1 param2=value2 ...
```

Parameters prefixed with `default..` (e.g. `default.optimizer=adam`), or 
parameters without prefix, are used for all classifiers. To specify 
a hyper-parameter for a specific classifier,  the name of the classifier 
may be used (e.g. `LinearConv1Model.number-of-convolutions=2`). In case 
the same classifier is used multiple times, 0-based indexing may be used to 
specify hyperparameters per classifier
(e.g `FullyConnectedModel[0].number-of-hidden-layers=2`).

When multiple classifiers are specified, the models are usually combined 
into 1 larger model, or are used in an ensemble. The simplest way of 
combining models is by combining them into 1 single network. This is 
the default behaviour when multiple classifiers are given. The method 
used to combine the models (default is concatenation) can be configured 
using the `--combination-strategy` argument.

Alternatively, the models can be combined using an ensemble strategy.
Currently, there are three supported ensembles: stacking, boosting, and 
voting.

Voting simply trains all the given classifiers separately, and uses a 
majority vote tot classify samples in the test set with them. This 
ensemble has no additional parameters.

The boosting ensemble takes in a single classifier, and performs boosting 
using the Adaboost algorithm. The number of boosting rounds can be specified
using the `--boosting-rounds` arguments.

Finally, there is the stacking ensemble. When performing stacking, the 
outputs of the base classifiers are combined into a new feature vector 
which is fed into a final meta classifier. In this case, the meta 
classifier is a deep learning model. The model can be specified using 
the `--stacking-meta-classifier` arguments. Hyper-parameters for the 
meta classifier can be given using `--stacking-meta-classifier-hyper-parameters`.
Note that the names of the hyper-parameters should not be prefixed in this case.

When using stacking, the program automatically attempts to determine a way 
to convert classifier outputs to new feature vectors. Specifically, there
are the following rules:

1) For detection, outputs are always concatenated into a single vector 
2) For Classification3Simplified and Classification8, outputs are combined
    into a matrix if the meta model supports matrix inputs. Otherwise,
    the argmax is taken from every model output, and these are concatenated 
    into a vector.
3) For Classification3, the outputs are either combined into a matrix 
    if the meta classifiers supports this. Otherwise, the output vectors
   (consisting of three Booleans) are interpreted as binary numbers 
    and concatenated into a single vector.

The `--stacking-no-matrix` option can be used to disable the use 
of matrices when generating new feature vectors. The `--stacking-use-concat`
parameter can be used to force concatenation, but this option is buggy
and should not be used. 

When `--test-separately` is given, all given classifiers are not combined,
but trained and evaluated separately. 

# Training Process Arguments

The following parameters have an influence on the training process. 
We will cover those in this section.

## Configuring Test Runs 

In the most basic evaluation setup, the classifier is trained and tested
once. The dataset given as argument is split into a training set, 
a validation set, and a testing set. The validation set and testing sets 
have the same size. This size is determined using the `--split-size` parameter.
This parameter is a float denoting the size of the testing and validation set.
If 0.2 is given, 20% of the data is used for testing, 20% for validation, 
and 60% for training.

If in stead, one wants to perform k-fold cross-validation, the `--k-cross <k>`
argument must be given. By default, this is a 10*9 nested cross validation.
The outer cross validation splits the dataset in 10 folds. 1 is the test set,
and then another cross validation is done with the "inner" folds, where 1 
is validation set and the others are training set. For k = 10, this requires 
a total of `10 * 9 = 90` runs. 

In order to reduce the amount of runs, the `--quick-cross` flag can be
supplied. When this flag is given, the dataset is split into `k` folds
(as specified by the `--k-cross` argument), where 1 fold is used as testing
set, 1 as validation set, and the remaining fold as training set. For k = 10, 
this only requires 9 runs.

When using `--k-cross` in combination with `--quick-cross`, a special flag 
may be given: `--cross-is-cross-project`. When using this flag, the dataset 
is split into folds based on the different projects in the dataset. 1 by 1,
each project is used as the test set, while the remaining projects are 
combined, shuffled, and used to generate training and validation sets.

A different variant which can be used in combination with `--quick-cross`,
is the `--test-study <study>` argument. When using this argument, an input 
file containing multiple studies
(e.g. `datasets/deep_learning/issuedata/EBSE_BHAT_issues_formatting-keep.json`)
must be used for the `--file` argument. When running in this mode, one study 
is used as the test set (the one given as the argument), and the other is used 
for the training and validation set (in a k-fold cross manner).

`--test-project` is an unused legacy option which has been replaced by 
`--cross-is-cross-project`.

## Miscellaneous Dataset Manipulation

The `--max-train <M>` argument can be used to limit the size of the training set,
in any mode. The goal of this is to evaluate how the classifiers perform when 
having access to less data, without having to modify the input `--file`.

Some datasets may have imbalanced data. The `--class-balancer` can be used 
to help remedy this. Possible values which may be given are "class-weight" and 
"upsample". Class limiting can also be done, but this must be done by 
specifying `class-limit=` using the `--params` argument.

## The training process
`--batch-size` can be used to configure the batch size used during the
training process.

The `--epochs` argument can be used to configure for how long
(how many epochs) the 
deep learning classifier will be trained. 
It is also possible to stop indefinitely train the classifier, until 
some objective attribute has stopped improving. 
This is called early stopping, which can be enabled using the 
`--use-early-stopping` flag. The objective attribute may be specified 
using `--early-stopping-attribute`. The training will be stopped 
if the value of the attribute has not improved by some minimum delta 
for some amount of epochs. The minimum delta can be set using 
`--early-stopping-min-delta`, while the amount of epochs can be set 
using `--early-stopping-patience`. 

# Miscellaneous Parameters 
The `--peregrine` flag is used to enable running on the Peregrine 
HPC cluster of the University of Groningen. This flag is untested. 

# Example Argument
Below is an example CLI argument for running a model:

```shell 
python __main__.py run FullyConnectedModel \
    --input-mode BOWFrequency \
    --output-mode Detection \
    --file datasets/deep_learning/issuedata/EBSE_formatting-keep.json \
    --params max-len=400 \
    --hyper-params \
        number-of-hidden-layers=1 \
        hidden-layer-1-size=16 \
        optimizer=adam \
        loss=crossentropy \
    --k-cross 10 \
    --quick-cross \
    --use-early-stopping \
    --early-stopping-patience 30
```

# Result Files 
After the deep learning manager is done running, a file 
`most_recent_run.txt` is generated. This file contains the name
of the result JSON file containing information about the test run that 
has just finished. 

The result files are fairly self-explanatory. They contain information 
about how labels map to classes, the ground truth of the test set,
the predictions of the trained models, and various performance 
metrics collected on the training, validation, and test sets. 
For almost all models, the metrics are collected every epoch. 
The only exception is the voting classifier, where the results are 
collected only after the training.

The result files may also contain an `early-stopping-settings` entry,
that can be used to determine whether the training of the model was 
done using early stopping. If it is, the final entry in the list 
for every metric is the value of that metric that the model achieved 
its best performance.
