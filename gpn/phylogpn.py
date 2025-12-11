from transformers import AutoModel, AutoTokenizer

checkpoint = "songlab/PhyloGPN"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True)
