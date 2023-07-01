# Command: `make-features`

---

The `make-features` command is used to generate feature vectors.

When running `python __main__.py make-features -h`, you get the following output:

```
usage: __main__.py make-features [-h] [--input-mode INPUT_MODE [INPUT_MODE ...]] [--output-mode OUTPUT_MODE] [--file FILE] [--params PARAMS [PARAMS ...]] [--ontology-classes ONTOLOGY_CLASSES]
                                 [--apply-ontology-classes]

options:
  -h, --help            show this help message and exit
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

The `--input-mode` argument is used to specify the input mode.
The input mode is also known as the feature vector generator. It 
is the method used to generate feature vectors from the text. 
The possible options can be seen using the [list](./list.md) command.

The `--output-mode` command is used to specify the output mode. 
The output mode is also known as the [task](./basics/tasks.md).

`--file` is a file containing the raw issue data. Example files can be 
found in the `datasets` folder.

`--ontology-classes` is a path to a file containing ontology classes,
obtained using `utility_scripts/compile_ontologies.py`. The program
will throw an error when it expects this parameter to be given --
though it should only be needed when running with `--apply-ontology-classes`
or using the `OntologyFeatures` generator.

The `--apply-ontology-classes` parameter specifies that the issue text
must be simplified using ontology classes. Words occurring in the 
given ontology classes will in stead be replaced with the class names.

`--params` is a list of `key=value` items used to specify parameters for the 
feature generator. See the docs of the [run](./run.md) command for 
more information; `--params` works in exactly the same way as 
`--hyper-params`, including prefixing mechanics in case multiple 
feature generators are used (in the `run` command).