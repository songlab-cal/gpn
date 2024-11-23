import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModel

import gpn.model
from gpn.data import Tokenizer, ReverseComplementer


def euclidean_distance(embed_ref, embed_alt):
    B = len(embed_ref)
    return F.pairwise_distance(embed_ref.reshape(B, -1), embed_alt.reshape(B, -1))


def euclidean_distances(embed_ref, embed_alt):
    return torch.linalg.norm(embed_ref - embed_alt, dim=1)


def inner_product(embed_ref, embed_alt):
    return (embed_ref * embed_alt).sum(dim=(1, 2))


def inner_products(embed_ref, embed_alt):
    return (embed_ref * embed_alt).sum(dim=1)


def cosine_distance(embed_ref, embed_alt):
    B = len(embed_ref)
    return 1 - F.cosine_similarity(embed_ref.reshape(B, -1), embed_alt.reshape(B, -1), dim=1)


def cosine_distances(embed_ref, embed_alt):
    return 1 - F.cosine_similarity(embed_ref, embed_alt, dim=1)


class VEPEmbeddings(torch.nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_path)

    def get_embedding(self, input_ids, aux_features):
        return self.model(
            input_ids=input_ids, aux_features=aux_features,
        ).last_hidden_state

    def get_scores(self, input_ids_ref, aux_features_ref, input_ids_alt, aux_features_alt):
        embed_ref = self.get_embedding(input_ids_ref, aux_features_ref)
        embed_alt = self.get_embedding(input_ids_alt, aux_features_alt)
        return torch.cat((
            torch.unsqueeze(euclidean_distance(embed_ref, embed_alt), 1),
            torch.unsqueeze(inner_product(embed_ref, embed_alt), 1),
            torch.unsqueeze(cosine_distance(embed_ref, embed_alt), 1),
            euclidean_distances(embed_ref, embed_alt),
            inner_products(embed_ref, embed_alt),
            cosine_distances(embed_ref, embed_alt),
        ), dim=1)

    def forward(
        self,
        input_ids_ref_fwd=None,
        aux_features_ref_fwd=None,
        input_ids_alt_fwd=None,
        aux_features_alt_fwd=None,
        input_ids_ref_rev=None,
        aux_features_ref_rev=None,
        input_ids_alt_rev=None,
        aux_features_alt_rev=None,
    ):
        fwd = self.get_scores(
            input_ids_ref_fwd, aux_features_ref_fwd, input_ids_alt_fwd, aux_features_alt_fwd,
        )
        rev = self.get_scores(
            input_ids_ref_rev, aux_features_ref_rev, input_ids_alt_rev, aux_features_alt_rev,
        )
        return (fwd + rev) / 2


class VEPEmbeddingsInference(object):
    def __init__(self, model_path, genome_msa, window_size, disable_aux_features=False):
        self.model = VEPEmbeddings(model_path)
        self.genome_msa = genome_msa
        self.window_size = window_size
        self.disable_aux_features = disable_aux_features
        self.tokenizer = Tokenizer()
        self.reverse_complementer = ReverseComplementer()

    def tokenize_function(self, V):
        # we convert from 1-based coordinate (standard in VCF) to
        # 0-based, to use with GenomeMSA
        chrom = np.array(V["chrom"])
        pos = np.array(V["pos"]) - 1
        start = pos - self.window_size // 2
        end = pos + self.window_size // 2
        
        msa_fwd, msa_rev = self.genome_msa.get_msa_batch_fwd_rev(
            chrom,
            start,
            end,
            tokenize=True,
        )
        pos_fwd = self.window_size // 2
        pos_rev = pos_fwd - 1 if self.window_size % 2 == 0 else pos_fwd

        ref_fwd = np.array(
            [np.frombuffer(x.encode("ascii"), dtype="S1") for x in V["ref"]]
        )
        alt_fwd = np.array(
            [np.frombuffer(x.encode("ascii"), dtype="S1") for x in V["alt"]]
        )
        ref_rev = self.reverse_complementer(ref_fwd)
        alt_rev = self.reverse_complementer(alt_fwd)

        def prepare_output(msa, pos, ref, alt):
            ref, alt = self.tokenizer(ref.flatten()), self.tokenizer(alt.flatten())
            input_ids, aux_features = msa[:, :, 0], msa[:, :, 1:]
            assert (
                input_ids[:, pos] == ref
            ).all(), f"{input_ids[:, pos].tolist()}, {ref.tolist()}"
            input_ids_alt = input_ids.copy()
            input_ids_alt[:, pos] = alt
            input_ids = input_ids.astype(np.int64)
            input_ids_alt = input_ids_alt.astype(np.int64)
            return input_ids, aux_features, input_ids_alt, aux_features

        res = {}
        (
            res["input_ids_ref_fwd"],
            res["aux_features_ref_fwd"],
            res["input_ids_alt_fwd"],
            res["aux_features_alt_fwd"],
        ) = prepare_output(msa_fwd, pos_fwd, ref_fwd, alt_fwd)
        (
            res["input_ids_ref_rev"],
            res["aux_features_ref_rev"],
            res["input_ids_alt_rev"],
            res["aux_features_alt_rev"],
        ) = prepare_output(msa_rev, pos_rev, ref_rev, alt_rev)
        if self.disable_aux_features:
            del res["aux_features_ref_fwd"]
            del res["aux_features_alt_fwd"]
            del res["aux_features_ref_rev"]
            del res["aux_features_alt_rev"]
        return res

    def postprocess(self, pred):
        D = (pred.shape[1] // 3) - 1
        cols = (
            ["euclidean_distance", "inner_product", "cosine_distance"]
            + [f"euclidean_distance_{i}" for i in range(D)]
            + [f"inner_product_{i}" for i in range(D)]
            + [f"cosine_distance_{i}" for i in range(D)]
        )
        return pd.DataFrame(pred, columns=cols)
