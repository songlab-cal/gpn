import sentencepiece as spm
import sys


train_file = sys.argv[1]
vocab_size = int(sys.argv[2])
output_path = sys.argv[3]


spm.SentencePieceTrainer.train(
    input=train_file,
    model_prefix=output_path,
    vocab_size=vocab_size,
    num_threads=40,
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
