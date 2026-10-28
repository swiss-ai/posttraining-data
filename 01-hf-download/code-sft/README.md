# Coding SFT download

`resolve_catalog.py` pins a user-supplied manifest to dataset revisions and source
files. `download.py` streams the files through the stage-02 normalizer, preserving
source records and file-level checkpoints. `benchmarks.py` downloads separately
pinned evaluation references for stage 04; it does not add them to training input.

Use the [code-SFT recipe](../../examples/code-sft/README.md) for the environment,
manifest format and runnable commands. Run through its dispatcher so imports from
stages 02 and 04 are available. Download and evaluation output paths are CLI inputs.
