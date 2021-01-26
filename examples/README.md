# Examples

This directory contains example of how to use the functionality offered by the *mmlib*.

- *probe_store.py* - Creates and stores a probe summary of the training process of an GoogLeNet.
    - execution: `python probe_store.py --path <optional path to store probe summary>`
- *probe_load_compare.py* - Creates a probe summary of the training process of an GoogLeNet and compares it to a stored
  probe summary
    - execution: `python probe_load_compare.py --path <path to the already stored probe summary>`
    - note: To generate and store a probe summary to compare to use the *probe_store.py* script.
- *probe_example.py* - Show extensively how the probe functionality offered by the *mmlib* can be used to make the
  PyTroch implementation of GoogLeNet reproducible. It runs the following steps:
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
        - compares both models using the methods *blackbox_equal*, *whitebox_equal*, and *equal* offered by the *mmlib*.