import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForMaskedLM

import gpn.star.model
from gpn.data import Tokenizer, ReverseComplementer


class MLMforVEPModel(torch.nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.model = AutoModelForMaskedLM.from_pretrained(model_path)
        self.model.eval()
        
    def get_llr(self, input_ids, source_ids, target_species, pos, ref, alt):
        logits = self.model.forward(
            input_ids=input_ids, source_ids=source_ids, 
            target_species=target_species,
        ).logits
        logits = logits[torch.arange(len(pos)), pos, 0]
        logits_ref = logits[torch.arange(len(ref)), ref]
        logits_alt = logits[torch.arange(len(alt)), alt]
        llr = logits_alt - logits_ref
        return llr

    def forward(
        self,
        input_ids_fwd=None,
        source_ids_fwd=None,
        pos_fwd=None,
        ref_fwd=None,
        alt_fwd=None,
        input_ids_rev=None,
        source_ids_rev=None,
        pos_rev=None,
        ref_rev=None,
        alt_rev=None,
        target_species=None
    ):

        llr_fwd = self.get_llr(
            input_ids_fwd, source_ids_fwd, target_species, pos_fwd, ref_fwd, alt_fwd, 
        )
        llr_rev = self.get_llr(
            input_ids_rev, source_ids_rev, target_species, pos_rev, ref_rev, alt_rev,
        )
        llr = (llr_fwd + llr_rev) / 2
        return llr


class VEPInference(object):
    def __init__(self, model_path, genome_msa_list, window_size, disable_aux_features=False):
        self.model = MLMforVEPModel(model_path)
        self.genome_msa_list = genome_msa_list
        self.window_size = window_size
        self.disable_aux_features = disable_aux_features
        self.reverse_complementer = ReverseComplementer()
        self.tokenizer = Tokenizer()
        # self.clade_dict = self.model.model.model.phylo_info.clade_dict
        # print(self.clade_dict)
        # print('Max evol dist:', self.model.model.config.max_evol_dist)

    def tokenize_function(self, V):
        # we convert from 1-based coordinate (standard in VCF) to
        # 0-based, to use with GenomeMSA
        chrom = np.array(V["chrom"])
        pos = np.array(V["pos"]) - 1
        
        start = pos - self.window_size // 2
        end = pos + self.window_size // 2
        
        msa_fwd, msa_rev = zip(*[
            genome_msa.get_msa_batch_fwd_rev(chrom, start, end, tokenize=True)
            for genome_msa in self.genome_msa_list
        ])
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

            
            # subsample
            input_ids = msa[:, :, :1]
            
            # assert (
            #     input_ids[:, pos, 0] == ref
            # ).all(), f"{input_ids[:, pos, 0].tolist()}, {ref.tolist()}"
            if not (input_ids[:, pos, 0] == ref).all():
                print(f"ref genome and variant file unmatched: {input_ids[:, pos, 0].tolist()}, {ref.tolist()}")
            input_ids[:, pos, 0] = self.tokenizer.mask_token_id()
            # also mask position in source_ids in human clade
            msa[:, pos, 0] = self.tokenizer.mask_token_id()
            pos = np.full(input_ids.shape[0], pos)
            ref, alt = ref.astype(np.int64), alt.astype(np.int64)
        
            
            return input_ids, msa, pos, ref, alt

        res = {}
        (
            res["input_ids_fwd"],
            res["source_ids_fwd"],
            res["pos_fwd"],
            res["ref_fwd"],
            res["alt_fwd"],
        ) = prepare_output(msa_fwd, pos_fwd, ref_fwd, alt_fwd)
        (
            res["input_ids_rev"],
            res["source_ids_rev"],
            res["pos_rev"],
            res["ref_rev"],
            res["alt_rev"],
        ) = prepare_output(msa_rev, pos_rev, ref_rev, alt_rev)

        res["target_species"] = np.zeros((chrom.shape[0], 1), dtype=int)
            
        return res

    def postprocess(self, pred):
        return pd.DataFrame(pred, columns=["score"])
