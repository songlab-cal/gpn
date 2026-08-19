# Python API

The curated public API is deliberately small. Model architectures are loaded with
Transformers AutoClasses after explicit registration.

## Registration

```{eval-rst}
.. autofunction:: gpn.register_auto_classes
```

## Scientific scoring helpers

```{eval-rst}
.. autofunction:: gpn.scoring.nucleotide_probabilities

.. autofunction:: gpn.scoring.log_likelihood_ratio
```

## PhyloGPN classes

```{eval-rst}
.. autoclass:: gpn.phylogpn.PhyloGPNConfig
   :members:

.. autoclass:: gpn.phylogpn.PhyloGPNTokenizer
   :members: build_inputs_with_special_tokens, get_vocab

.. autoclass:: gpn.phylogpn.PhyloGPNModel
   :members: get_embeddings, forward
```

Internal training, model, and data modules are not automatically declared stable
merely because they are importable.
