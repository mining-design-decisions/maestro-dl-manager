# Deep Learning Tasks

---

The deep learning task, or output mode, is used to denote what kind 
of deep learning task the neural network will be trained for. 
We hava implemented the following tasks:

- Detection: determine whether an issue is architectural or not 
- Classification3: determine _all_ labels of issues, where labels are encoded 
    as Boolean vectors of length 3, where the first entry stands for executive, the
    second one for existence, and the last one for property
- Classification3Simplified: determine the _primary_ label of issues. 
    The primary label is the label of an issue with the highest importance.
    We have Executive > Property > Existence. Hence, an issue with labels 
    "Executive" and "Existence", the _only_ label of the issue would be "Executive"
- Classification8: All 8 combinations of (executive, existence, property) 
    one-hot encoded using a vector of length 8.