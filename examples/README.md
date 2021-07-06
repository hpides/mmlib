# Examples

This directory contains examples of how to use the functionality offered by the *mmlib*.

## Approaches to save and recover models
- to execute all examples we use a MongoDB, in all examples the MongoDB is started using docker
- if you don't have docker installed you have to either install it or slightly adjust the examples
- in `baseline_save.py` we provide an example of how to save and recover a model using the baseline approach
- for all other approaches we do not give explict examples and refer to our [test for the appraoches](../tests/save)


## Probing Tool

We provide some basic examples to see the different use cases of the probing tool

### Create a probe summary for a given model
- *probe_store.py* - Creates and stores a probe summary of the training process of a GoogLeNet.
- execution: `python probe_store.py --path <optional path to store probe summary>`
    
### Create new summary and compare to given one 
- *probe_load_compare.py* - Creates a probe summary of the training process of a GoogLeNet and compares it to a stored
  probe summary
- execution: `python probe_load_compare.py --path <path to the already stored probe summary>`
- note: To generate and store a probe summary to compare to use the *probe_store.py* script.
    
### Extensive example
- *probe_example.py* - Shows extensively how the probe functionality offered by the *mmlib* can be used to make the
  PyTorch implementation of GoogLeNet reproducible. It runs the following steps:
    - simple summary
        - creates a probe summary for the inference mode and prints the representation
    - probe inference
        - creates two instances of the same model
        - creates inference mode probe summaries (covering forward path) for them
        - compares the probe summaries
    - probe training
        - creates two instances of the same model
        - creates training mode probe summaries (covering forward and backward path)
        - compares the probe summaries
    - probe reproducible training
        - creates two instances of the same model
        - uses *set_deterministic* functionality offered by the *mmlib* to make the training process of both models
          reproducible
        - creates training mode probe summaries (covering forward and backward path)
        - compares the probe summaries
        - compares both models using the methods *blackbox_model_equal*, *whitebox_model_equal*, and *model_equal*
          offered by the *mmlib*.