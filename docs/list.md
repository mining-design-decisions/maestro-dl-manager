# Command: `list`

---

The `list` command is used to obtain the various options for:

- The input mode (feature vector generator)
- The output mode (deep learning task)
- The available classifiers.

``` 
usage: __main__.py list [-h] {classifiers,inputs,outputs}

positional arguments:
  {classifiers,inputs,outputs}
                        Possible categories to list options for.

options:
  -h, --help            show this help message and exit
```

Example:

```shell 
> python __main__.py list classifiers 
Available Classifiers:
	* Bert
	* FullyConnectedModel
	* LinearConv1Model
	* LinearRNNModel
	* NonlinearConv2Model
```