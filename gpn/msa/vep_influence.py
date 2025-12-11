import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM

import gpn.model
from gpn.data import Tokenizer, ReverseComplementer


class VEPInfluence(torch.nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.model = AutoModelForMaskedLM.from_pretrained(model_path)
        tokenizer = Tokenizer()
        self.vocab_start = tokenizer.nucleotide_token_id_start()
        self.vocab_end = tokenizer.nucleotide_token_id_end()

    def get_log_odds(self, input_ids, aux_features, pos):
        other_pos = torch.ones(input_ids.shape[1], dtype=torch.bool)
        other_pos[pos] = False
        logits = self.model(input_ids=input_ids, aux_features=aux_features).logits[
            :, other_pos
        ]
        logits = logits[:, :, self.vocab_start : self.vocab_end]
        probs = F.softmax(logits, dim=2)
        return torch.log(probs / (1 - probs))

    def get_score(
        self, input_ids_ref, aux_features_ref, input_ids_alt, aux_features_alt, pos
    ):
        log_odds_ref = self.get_log_odds(input_ids_ref, aux_features_ref, pos)
        log_odds_alt = self.get_log_odds(input_ids_alt, aux_features_alt, pos)
        res = torch.abs(log_odds_ref - log_odds_alt)
        res, _ = torch.max(res, dim=2)
        res = torch.mean(res, dim=1)
        return res

    def forward(
        self,
        input_ids_ref_fwd=None,
        aux_features_ref_fwd=None,
        input_ids_alt_fwd=None,
        aux_features_alt_fwd=None,
        pos_fwd=None,
        input_ids_ref_rev=None,
        aux_features_ref_rev=None,
        input_ids_alt_rev=None,
        aux_features_alt_rev=None,
        pos_rev=None,
    ):
        fwd = self.get_score(
            input_ids_ref_fwd,
            aux_features_ref_fwd,
            input_ids_alt_fwd,
            aux_features_alt_fwd,
            pos_fwd,
        )
        rev = self.get_score(
            input_ids_ref_rev,
            aux_features_ref_rev,
            input_ids_alt_rev,
            aux_features_alt_rev,
            pos_rev,
        )
        return (fwd + rev) / 2


class VEPInfluenceInference(object):
    def __init__(self, model_path, genome_msa, window_size, disable_aux_features=False):
        self.model = VEPInfluence(model_path)
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
        n = len(chrom)

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
            assert (input_ids[:, pos] == ref).all(), (
                f"{input_ids[:, pos].tolist()}, {ref.tolist()}"
            )
            input_ids_alt = input_ids.copy()
            input_ids_alt[:, pos] = alt
            input_ids = input_ids.astype(np.int64)
            input_ids_alt = input_ids_alt.astype(np.int64)
            pos = np.full(n, pos, dtype=np.int64)
            return input_ids, aux_features, input_ids_alt, aux_features, pos

        res = {}
        (
            res["input_ids_ref_fwd"],
            res["aux_features_ref_fwd"],
            res["input_ids_alt_fwd"],
            res["aux_features_alt_fwd"],
            res["pos_fwd"],
        ) = prepare_output(msa_fwd, pos_fwd, ref_fwd, alt_fwd)
        (
            res["input_ids_ref_rev"],
            res["aux_features_ref_rev"],
            res["input_ids_alt_rev"],
            res["aux_features_alt_rev"],
            res["pos_rev"],
        ) = prepare_output(msa_rev, pos_rev, ref_rev, alt_rev)
        if self.disable_aux_features:
            del res["aux_features_ref_fwd"]
            del res["aux_features_alt_fwd"]
            del res["aux_features_ref_rev"]
            del res["aux_features_alt_rev"]
        return res

    def postprocess(self, pred):
        return pd.DataFrame(pred, columns=["score"])
