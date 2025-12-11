import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForMaskedLM
import os

import gpn.star.model
from gpn.data import Tokenizer


class MLMforLogitsModel(torch.nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.model = AutoModelForMaskedLM.from_pretrained(model_path)
        self.model.eval()
        tokenizer = Tokenizer()
        self.id_a = tokenizer.vocab.index("A")
        self.id_c = tokenizer.vocab.index("C")
        self.id_g = tokenizer.vocab.index("G")
        self.id_t = tokenizer.vocab.index("T")

    def get_logits(self, input_ids, source_ids, target_species, pos):
        logits = self.model.forward(
            input_ids=input_ids,
            source_ids=source_ids,
            target_species=target_species,
        ).logits
        logits = logits[torch.arange(len(pos)), pos].squeeze(-2)
        return logits

    def forward(
        self,
        input_ids_fwd=None,
        source_ids_fwd=None,
        pos_fwd=None,
        input_ids_rev=None,
        source_ids_rev=None,
        pos_rev=None,
        target_species=None,
    ):
        id_a = self.id_a
        id_c = self.id_c
        id_g = self.id_g
        id_t = self.id_t
        logits_fwd = self.get_logits(
            input_ids_fwd, source_ids_fwd, target_species, pos_fwd
        )[:, [id_a, id_c, id_g, id_t]]
        logits_rev = self.get_logits(
            input_ids_rev, source_ids_rev, target_species, pos_rev
        )[:, [id_t, id_g, id_c, id_a]]
        return (logits_fwd + logits_rev) / 2


class LogitsInference(object):
    def __init__(
        self, model_path, genome_msa_list, window_size, disable_aux_features=False
    ):
        self.model = MLMforLogitsModel(model_path)
        self.genome_msa_list = genome_msa_list
        self.window_size = window_size
        self.disable_aux_features = disable_aux_features
        self.tokenizer = Tokenizer()

    def tokenize_function(self, V):
        # we convert from 1-based coordinate (standard in VCF) to
        # 0-based, to use with GenomeMSA
        chrom = np.array(V["chrom"])
        pos = np.array(V["pos"]) - 1
        start = pos - self.window_size // 2
        end = pos + self.window_size // 2

        msa_fwd, msa_rev = zip(
            *[
                genome_msa.get_msa_batch_fwd_rev(chrom, start, end, tokenize=True)
                for genome_msa in self.genome_msa_list
            ]
        )
        msa_fwd = np.concatenate(msa_fwd, axis=-1)
        msa_rev = np.concatenate(msa_rev, axis=-1)

        pos_fwd = self.window_size // 2
        pos_rev = pos_fwd - 1 if self.window_size % 2 == 0 else pos_fwd

        def prepare_output(msa, pos):
            input_ids = msa[:, :, :1]
            input_ids[:, pos] = self.tokenizer.mask_token_id()
            input_ids = input_ids.astype(np.int64)
            pos = np.full(len(input_ids), pos)
            return input_ids, msa, pos

        res = {}
        (
            res["input_ids_fwd"],
            res["source_ids_fwd"],
            res["pos_fwd"],
        ) = prepare_output(msa_fwd, pos_fwd)
        (
            res["input_ids_rev"],
            res["source_ids_rev"],
            res["pos_rev"],
        ) = prepare_output(msa_rev, pos_rev)

        res["target_species"] = np.zeros((chrom.shape[0], 1), dtype=int)

        return res

    def postprocess(self, pred):
        return pd.DataFrame(pred, columns=list("ACGT"))
