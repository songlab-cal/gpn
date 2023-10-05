import numpy as np
import pandas as pd
import torch
from transformers import AutoModel

import gpn.model


class ModelCenterEmbedding(torch.nn.Module):
    def __init__(self, model_path, center_window_size):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_path)
        self.center_window_size = center_window_size

    def get_center_embedding(self, input_ids, aux_features=None):
        embedding = self.model.forward(
            input_ids=input_ids, aux_features=aux_features
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
        aux_features_fwd=None,
        aux_features_rev=None,
    ):
        embedding_fwd = self.get_center_embedding(input_ids_fwd, aux_features_fwd)
        embedding_rev = self.get_center_embedding(input_ids_rev, aux_features_rev)
        embedding = (embedding_fwd + embedding_rev) / 2
        return embedding


class EmbeddingInference(object):
    def __init__(
        self,
        model_path,
        genome_msa,
        window_size,
        disable_aux_features=False,
        center_window_size=100,
    ):
        self.model = ModelCenterEmbedding(model_path, center_window_size)
        self.genome_msa = genome_msa
        self.window_size = window_size
        self.disable_aux_features = disable_aux_features

    def tokenize_function(self, V):
        chrom = np.array(V["chrom"])
        start = np.array(V["start"])
        end = np.array(V["end"])
        msa_fwd, msa_rev = self.genome_msa.get_msa_batch_fwd_rev(
            chrom,
            start,
            end,
            tokenize=True,
        )

        def prepare_output(msa):
            input_ids, aux_features = msa[:, :, 0], msa[:, :, 1:]
            input_ids = input_ids.astype(np.int64)
            return input_ids, aux_features

        res = {}
        (
            res["input_ids_fwd"],
            res["aux_features_fwd"],
        ) = prepare_output(msa_fwd)
        (
            res["input_ids_rev"],
            res["aux_features_rev"],
        ) = prepare_output(msa_rev)
        if self.disable_aux_features:
            del res["aux_features_fwd"]
            del res["aux_features_rev"]
        return res

    def postprocess(self, pred):
        cols = [f"embedding_{i}" for i in range(pred.shape[1])]
        return pd.DataFrame(pred, columns=cols)
