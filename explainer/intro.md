---
date: 2025-04-15
authors: Siddharth Setlur
affiliations: University of Edinburgh
hide_search: false
hide_title_block: false
numbering:
  code: true
  math: true
  headings: true
export:
  - format: pdf
    template: lapreprint

---


# How to Build "Large" Codebases 
## Motivation
I've recently been trying to reproduce some results in *mechanistic interpretability* in the hopes of building upon some of these results using tools from topology/geometry and doing so has made me realize that my janky coding just won't cut it anymore. I come from a math background, so coding for me has predominantly been coding Jupyter notebooks and perhaps the odd standalone script here and there to train a neural network or test topological loss functions. While it wasn't necessarily the nicest or more efficient code, it was often enough for my purposes of testing ideas. I've recently been thinking of ways one could apply tools from geometry/topology to problems in mechanistic interpretability, and attempting to design experiments with this code style has proven impossible for multiple reasons (not in order of importance):

1. The amount of data that one needs to process to train things like sparse autoencoders means the single script approach and hope for the best does not work. We need ways of sharding data to process in chunks, parallelization strats, and means of keeping track of the dozens of hyperparameters involved. 
2. Choosing the "right" hyperparameters is no longer a matter of running for a few iterations, checking if the training loss/validation loss is decreasing and adapting accordingly. We need a means of searching over the hyperparameter space in a meaningful manner and keeping track of results. 
3. We need our models and the analysis we do on these models to be reproducible. 
4. We need easy to understand functions that we can call to analyze the sparse autoencoders and the feature manifolds that we obtain using these models. 

In this post, I want to document myself stumbling my way to understanding what I imagine is second nature to software engineers. In particular, the problem I'm currently facing is training a sparse autoencoder on ESM2 following @simon_interplm_2025 and the [associated (and extremely well-written) codebase](https://github.com/ElanaPearl/InterPLM/tree/main). We first attempted to use the pretained SAEs provided as part of the paper, but soon realized that it would be beneficial to train our own SAEs to extract the features we were interested in (perhaps an SAE with a different choice of hyperparameter would have more features that activate on the geometric concepts we are interested in). Now, the best practice here would be to fork [the repo]((https://github.com/ElanaPearl/InterPLM/tree/main)) and attempt to edit the code as needed for our purposes, but despite the fact that this is probably one of the best written and well-documented codebases I've seen, I was struggling to decipher some of the more sophisticated object-oriented concepts used here and how everything fits together. So instead, I decided to copy over (what I thought) were the essential pieces needed to train a SAE on a PLM. The plan is to go through the code, explain concepts that I hadn't encountered before, and document the edits I make to create a training script that accomplishes the objectives listed above. 

## Project structure
Let's first get a high-level view of the project structure. If like me, you're not used to modular projects it's a bit hard to understand what each part of the tree below does. 
 ```
 .

├── environment.yml
├── README.MD
├── proteinlens
     ├── __init__.py
│   ├── embedders
│   │   ├── base.py
│   │   ├── esm.py
│   │   ├── __init__.py
│   │   └── README.md
│   ├── sae
│   │   ├── dictionary.py
│   │   ├── inference.py
│   │   ├── __init__.py
│   │   ├── intervention.py
│   │   ├── normalize.py
│   ├── train
│   │   ├── checkpoint_manager.py
│   │   ├── configs.py
│   │   ├── data_loader.py
│   │   ├── evaluation.py
│   │   ├── fidelity.py
│   │   ├── trainers
│   │   │   ├── base_trainer.py
│   │   │   ├── common.py
│   │   │   ├── __init__.py
│   │   │   ├── jump_relu.py
│   │   │   ├── matryoshka_batch_top_k.py
│   │   │   └── relu.py
│   │   ├── training_run.py
│   │   └── wandb_manager.py
│   └── utils.py
├── scripts
│   ├── collect_feature_activations.py
│   ├── embed_annotations.py
│   ├── evaluate_sae.py
│   ├── extract_embeddings.py
│   ├── shard_fasta.py
│   └── subset_fasta.py
├── setup.py
```


Let's tackle the tree from top to bottom. The first thing we see is an `environment.yml` which specifies the packages we need in order to run everything and allows us to generate an appropriate conda environment. The only other file in the top-level directory is `setup.py` which simply allows us to make `proteinlens` a package that can be installed using `pip install -e . ` and enables us to write things like `import proteinlens`. The actual code for the package is contained in the `proteinlens`, and it is important that the name specified in `setup.py` matches the name of this subdirectory.  The first file we see in the `proteinlens` subdirectory (or any of the subdirectories of `proteinlens` for that matter) is the `__init__` file, which in general make the stuff inside the directory importable. For example, the `__init__.py` in the top level allows us to do things like `from proteinlens.utils import get_device`. Each subdirectory has its own `__init__.py` file that enables things like `from proteinlens.sae.embedders.esm import embed_single_sequence`. 

Each subdirectory of `proteinlens` implements a different functionality primarily by implementing classes. For example, the subdirectory `embedders` which contains the code that takes batches of protein sequences, feeds it to a PLM and returns the embeddings of these sequences at various layers of the PLM. Next up is the `sae` directory that implements the sparse autoencoder and associated functions like normalization, intervention (ablations), and inference. The `train` subdirectory defines training routines for the SAE variants defined in `sae` and some helper functions. 
