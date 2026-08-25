# Python API

This is the supported model-facing Python surface: explicit AutoClass
registration, scientific scoring helpers, principal configurations and model
heads, the PhyloGPN tokenizer, and genomic data readers. CLI dispatch, training
plumbing, checkpoint internals, and architecture building blocks remain
documented in source but are not part of the public stability contract.

This curation keeps the reference useful and avoids turning every importable
implementation detail into a compatibility promise. The trade-off is that some
advanced helpers are discoverable only in source; they can be promoted here when
the project deliberately supports direct use.

## Registration

```{eval-rst}
.. autofunction:: gpn.register_auto_classes
```

## Scientific scoring

```{eval-rst}
.. autofunction:: gpn.scoring.nucleotide_probabilities

.. autofunction:: gpn.scoring.log_likelihood_ratio
```

## GPN

```{eval-rst}
.. autoclass:: gpn.ss.model.GPNConfig

.. autoclass:: gpn.ss.model.GPNModel
   :members: forward

.. autoclass:: gpn.ss.model.GPNForMaskedLM
   :members: forward

.. autoclass:: gpn.ss.model.GPNForSequenceClassification
   :members: forward

.. autoclass:: gpn.ss.model.GPNForTokenClassification
   :members: forward

.. autoclass:: gpn.ss.model.ConvNetConfig

.. autoclass:: gpn.ss.model.ConvNetModel
   :members: forward

.. autoclass:: gpn.ss.model.ConvNetForMaskedLM
   :members: forward

.. autoclass:: gpn.ss.model.ConvNetForSequenceClassification
   :members: forward

.. autoclass:: gpn.ss.data.Genome
   :members:
```

## GPN-MSA

```{eval-rst}
.. autoclass:: gpn.msa.model.GPNMSAConfig

.. autoclass:: gpn.msa.model.GPNMSAModel
   :members: forward

.. autoclass:: gpn.msa.model.GPNMSAForMaskedLM
   :members: forward

.. autoclass:: gpn.msa.data.GenomeMSA
   :members:
```

## PhyloGPN

```{eval-rst}
.. autoclass:: gpn.phylo.model.PhyloGPNConfig

.. autoclass:: gpn.phylo.model.PhyloGPNTokenizer
   :members: build_inputs_with_special_tokens, get_vocab

.. autoclass:: gpn.phylo.model.PhyloGPNModel
   :members: get_embeddings, forward
```

## GPN-Star

```{eval-rst}
.. autoclass:: gpn.star.model.GPNStarConfig

.. autoclass:: gpn.star.model.GPNStarModel
   :members: forward

.. autoclass:: gpn.star.model.GPNStarForMaskedLM
   :members: forward

.. autoclass:: gpn.star.data.GenomeMSA
   :members:
```
