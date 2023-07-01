# Command: `hyperparams`

---

The `hyperparams` command is used to list the possible hyper-parameters 
for a given classifier. A list of classifiers can be obtained using the 
[list](./list.md) command.

Usage: `python __main__.py hyperparams [CLASSIFIER]`

Example:
``` 
> python __main__.py hyperparams FullyConnectedModel
Hyper-parameters for FullyConnectedModel:
	* hidden_layer_1_size -- [min, max] = [8, 128] -- default = 64
	* hidden_layer_2_size -- [min, max] = [8, 128] -- default = 32
	* hidden_layer_3_size -- [min, max] = [8, 128] -- default = 16
	* number_of_hidden_layers -- [min, max] = [0, 3] -- default = 1
	* use_trainable_embedding -- [min, max] = [False, True] -- default = False
```

Note: The hyperparameters displayed here use underscores (_), but hypens (-)
should be used when using them in the CLI.