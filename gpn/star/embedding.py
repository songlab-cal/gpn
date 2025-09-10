import numpy as np
import pandas as pd
import torch
from transformers import AutoModel

import gpn.star.model


class ModelCenterEmbedding(torch.nn.Module):
    def __init__(self, model_path, center_window_size):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_path)
        self.center_window_size = center_window_size

    def get_center_embedding(self, input_ids, source_ids, target_species):
        embedding = self.model.forward(
            input_ids=input_ids, source_ids=source_ids, target_species=target_species,
        ).last_hidden_state
        center = embedding.shape[1] // 2
        left = center - self.center_window_size // 2
        right = center + self.center_window_size // 2
        embedding = embedding[:, left:right]
        embedding = embedding.mean(axis=1)
        return embedding

    def forward(
        self,
        input_ids_fwd=None,
        input_ids_rev=None,
        source_ids_fwd=None,
        source_ids_rev=None,
        target_species=None,
    ):
        embedding_fwd = self.get_center_embedding(input_ids_fwd, source_ids_fwd, target_species)
        embedding_rev = self.get_center_embedding(input_ids_rev, source_ids_rev, target_species)
        embedding = (embedding_fwd + embedding_rev) / 2
        return embedding


class EmbeddingInference(object):
    def __init__(
        self,
        model_path,
        genome_msa_list,
        window_size,
        disable_aux_features=False,
        center_window_size=100,
    ):
        self.model = ModelCenterEmbedding(model_path, center_window_size)
        self.genome_msa_list = genome_msa_list
        self.window_size = window_size
        self.disable_aux_features = disable_aux_features
        # self.clade_dict = self.model.model.phylo_info.clade_dict
        # print(self.clade_dict)
        # print('Max evol dist:', self.model.model.config.max_evol_dist)

    def tokenize_function(self, V):
        chrom = np.array(V["chrom"])
        start = np.array(V["start"])
        end = np.array(V["end"])
        
        msa_fwd, msa_rev = zip(*[
            genome_msa.get_msa_batch_fwd_rev(chrom, start, end, tokenize=True)
            for genome_msa in self.genome_msa_list
        ])
        msa_fwd = np.concatenate(msa_fwd, axis=-1)
        msa_rev = np.concatenate(msa_rev, axis=-1)

        def prepare_output(msa):
            input_ids = msa[:, :, :1]
            input_ids = input_ids.astype(np.int64)
            msa[:, pos, 0] = self.tokenizer.mask_token_id()
            return input_ids, msa

        res = {}
        (
            res["input_ids_fwd"],
            res["source_ids_fwd"],
        ) = prepare_output(msa_fwd)
        (
            res["input_ids_rev"],
            res["source_ids_rev"],
        ) = prepare_output(msa_rev)
        
        res["target_species"] = np.zeros((chrom.shape[0], 1), dtype=int)
        
        if self.disable_aux_features:
            del res["source_ids_fwd"]
            del res["source_ids_rev"]
        return res

    def postprocess(self, pred):
        cols = [f"embedding_{i}" for i in range(pred.shape[1])]
        return pd.DataFrame(pred, columns=cols)
