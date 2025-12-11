import numpy as np
import pandas as pd
import torch
from transformers import AutoModel

import gpn.star.model
from gpn.data import Tokenizer, ReverseComplementer


class VEPEmbedding(torch.nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_path)

    def get_embedding(self, input_ids, source_ids, target_species):
        return self.model(
            input_ids=input_ids,
            source_ids=source_ids,
            target_species=target_species,
        ).last_hidden_state

    def get_scores(
        self,
        input_ids_ref,
        source_ids_ref,
        input_ids_alt,
        source_ids_alt,
        target_species,
    ):
        embedding_ref = self.get_embedding(
            input_ids_ref, source_ids_ref, target_species
        )  # (B, L, 1, H)
        embedding_alt = self.get_embedding(
            input_ids_alt, source_ids_alt, target_species
        )
        return (embedding_ref * embedding_alt).sum(dim=1).squeeze(1)

    def forward(
        self,
        input_ids_ref_fwd=None,
        source_ids_ref_fwd=None,
        input_ids_alt_fwd=None,
        source_ids_alt_fwd=None,
        input_ids_ref_rev=None,
        source_ids_ref_rev=None,
        input_ids_alt_rev=None,
        source_ids_alt_rev=None,
        target_species=None,
    ):
        fwd = self.get_scores(
            input_ids_ref_fwd,
            source_ids_ref_fwd,
            input_ids_alt_fwd,
            source_ids_alt_fwd,
            target_species,
        )
        rev = self.get_scores(
            input_ids_ref_rev,
            source_ids_ref_rev,
            input_ids_alt_rev,
            source_ids_alt_rev,
            target_species,
        )
        return (fwd + rev) / 2


class VEPEmbeddingInference(object):
    def __init__(
        self, model_path, genome_msa_list, window_size, disable_aux_features=False
    ):
        self.model = VEPEmbedding(model_path)
        self.genome_msa_list = genome_msa_list
        self.window_size = window_size
        self.disable_aux_features = disable_aux_features
        self.reverse_complementer = ReverseComplementer()
        self.tokenizer = Tokenizer()
        # self.clade_dict = self.model.model.phylo_info.clade_dict
        # print(self.clade_dict)
        # print('Max evol dist:', self.model.model.config.max_evol_dist)

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
            input_ids = msa[:, :, :1]
            # assert (
            #     input_ids[:, pos, 0] == ref
            # ).all(), f"{input_ids[:, pos, 0].tolist()}, {ref.tolist()}"
            if not (input_ids[:, pos, 0] == ref).all():
                print(
                    f"ref genome and variant file unmatched: {input_ids[:, pos, 0].tolist()}, {ref.tolist()}"
                )
                input_ids[:, pos, 0] = ref
            input_ids_alt = input_ids.copy()
            input_ids_alt[:, pos, 0] = alt
            input_ids = input_ids.astype(np.int64)
            input_ids_alt = input_ids_alt.astype(np.int64)
            msa[:, pos, 0] = self.tokenizer.mask_token_id()
            return input_ids, msa, input_ids_alt, msa

        res = {}
        (
            res["input_ids_ref_fwd"],
            res["source_ids_ref_fwd"],
            res["input_ids_alt_fwd"],
            res["source_ids_alt_fwd"],
        ) = prepare_output(msa_fwd, pos_fwd, ref_fwd, alt_fwd)
        (
            res["input_ids_ref_rev"],
            res["source_ids_ref_rev"],
            res["input_ids_alt_rev"],
            res["source_ids_alt_rev"],
        ) = prepare_output(msa_rev, pos_rev, ref_rev, alt_rev)

        res["target_species"] = np.zeros((chrom.shape[0], 1), dtype=int)

        if self.disable_aux_features:
            del res["source_ids_ref_fwd"]
            del res["source_ids_alt_fwd"]
            del res["source_ids_ref_rev"]
            del res["source_ids_alt_rev"]

        return res

    def postprocess(self, pred):
        # print(pred.shape)
        cols = [f"embedding_{i}" for i in range(pred.shape[1])]
        return pd.DataFrame(pred, columns=cols)
