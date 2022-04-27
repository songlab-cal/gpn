from tokenizers import decoders, models, normalizers, pre_tokenizers, processors, trainers, Tokenizer
from transformers import FNetTokenizer, AlbertTokenizer
import sentencepiece as spm
import sys

seq = "ATAAACATATCATAAATAAGATCAATATTAATAAAATAAATAGTTTTTTTTACGGGACGGATTGGCGGGACGAGTTTAGCAGGACGTAACTTAATAACAATTGTAAACTATAAAATAAAAATATTTTATAGATAGATACAATTTGCAAACTTTTATATATACTAACTTAAAAAAAAAATATTGTCCCCTGCGGTATAAGACGGGTTAAAAATCTAGTTGTTATTATTAAAGGAAATAAAATATCCTCATAAAACAATTTGTTGTAATCTATCTTTGGGCTAATGTTCTTATCCTACAAGACGAACCCTGACCGTATTCGTCGTAGAAAAAAAATTGCTTCGATCCCATCATTGAGTTCAATAATCGGCGCACAAAGGCCGATTCATAAAAACTCTAGGCCCATTAAAGTAAAGCCCATTCTCAACCCTATCCAGTCTCCCTGTATATATATATTTACGACACCAACCCAGCGTTGATATTTAATTTTCTTCAGTCAGAGATTTCGAAACCCTAGTCGATTTCGAGATCCAACTAACTCTGCTCCTTATCTCAGGTAAAATTCTCGCTCGAGAACTCAATTGCTTATCCAAAGTTCCAACTGAAGATGCTTTCCTACTGAATCTTAGGTTAATGTTTTGGATTTGGAATCTTACCCGAAATTTCTCTGCAGCTTGTTGAATTTGCGAAGTATGGGAGACGCTAGAGACAACGAAGCCTACGAGGAGGAGCTCTTGGACTATGAAGAAGAAGACGAGAAGGTCCCAGATTCTGGAAACAAAGTTAACGGCGAAGCTGTGAAAAAGTGAGTTTTATGGTTTCCTCGATATGTTTCATGTATACTACTGTGTGTTTAAATTTGTCGATTCTTAGATTACTACTTGATAACAAGTAGCAGTATGTGTTTAATTAGTTGCTTAACATATAACAATTGACTGAGTTCTTCATTGCTATAATTCCTGAAACCCACCCAATATTAGACTGTCGTGTGTTTCTCATATTG"

sp_model_kwargs = dict(enable_sampling=True, nbest_size=-1, alpha=1.5)

#sp = spm.SentencePieceProcessor(**sp_model_kwargs)
#sp.Load(sys.argv[1])
#encoded_seq = sp.encode(seq)
#print(len(seq), "->", len(encoded_seq))


tokenizer = AlbertTokenizer(
    vocab_file=sys.argv[1],
    bos_token="[CLS]",
    eos_token="[SEP]",
    unk_token="[UNK]",
    pad_token="[PAD]",
    mask_token="[MASK]",
    extra_ids=0,
    sp_model_kwargs=sp_model_kwargs,
    do_lower_case=True,
)
tokenizer.save_pretrained("tokenizer_test")
#cls_token_id = tokenizer.token_to_id("[CLS]")
#sep_token_id = tokenizer.token_to_id("[SEP]")
#cls_token_id = tokenizer.get_vocab()["[CLS]"]
#sep_token_id = tokenizer.get_vocab()["[SEP]"]
#tokenizer.post_processor = processors.TemplateProcessing(
#    single="[CLS]:0 $A:0 [SEP]:0",
#    pair="[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
#    special_tokens=[
#        ("[CLS]", cls_token_id),
#        ("[SEP]", sep_token_id),
#    ],
#)
print(len(tokenizer))
print(tokenizer.decode(tokenizer("ACGTTTGGGCA")["input_ids"]))
print(tokenizer("ACGTTTCA"))
