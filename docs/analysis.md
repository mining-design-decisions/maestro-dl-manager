Command: `run_analysis`

---

The `run_analysis` command has a number of sub-commands which can be 
used to analyze the results of the `run` command. The sub-commands are 
undocumented here, because the commands are unstable both in nature 
and interface.
However, information can still be found in the 
CLI itself using the syntax `python run_analysis <sub-command> -h`.

List of available sub-commands:

``` 
usage: __main__.py run_analysis [-h] {summarize,plot,compare,plot-attributes,confusion,compare-stats} ...

options:
  -h, --help            show this help message and exit

Subsub-commands:
  {summarize,plot,compare,plot-attributes,confusion,compare-stats}
    summarize           display the results for a single run
    plot                Plot various metrics in a plot
    compare             Compare various file through sorting
    plot-attributes     Plot an attribute across all folds.
    confusion           Plot the confusion matrix of the given file
    compare-stats       Test statistical significance of difference between k-fold runs.
```