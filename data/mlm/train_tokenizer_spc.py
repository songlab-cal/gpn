import sentencepiece as spm
import sys
from transformers import AlbertTokenizer


train_file = sys.argv[1]
vocab_size = int(sys.argv[2])
output_path = sys.argv[3]


spm.SentencePieceTrainer.train(
    input=train_file,
    model_prefix=output_path,
    vocab_size=vocab_size,
    num_threads=32,
    seed_sentencepiece_size=50000,
    add_dummy_prefix=False,
    bos_piece="[CLS]",
    bos_id=0,
    eos_piece="[SEP]",
    eos_id=1,
    unk_piece="[UNK]",
    unk_id=2,
    pad_piece="[PAD]",
    pad_id=3,
    user_defined_symbols="[MASK]",
)

sp_model_kwargs = dict(enable_sampling=True, nbest_size=-1, alpha=1.5)
tokenizer = AlbertTokenizer(
    vocab_file=output_path + ".model",
    bos_token="[CLS]",
    eos_token="[SEP]",
    unk_token="[UNK]",
    pad_token="[PAD]",
    mask_token="[MASK]",
    extra_ids=0,
    sp_model_kwargs=sp_model_kwargs,
    do_lower_case=True,
)
tokenizer.save_pretrained(output_path)
