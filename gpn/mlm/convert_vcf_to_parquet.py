import pandas as pd


input_path = "example_annotated.vcf.gz"
output_path = "example_annotated.parquet"

vcf = pd.read_csv(input_path, sep="\t", comment="#", header=None, names=["chromosome",  "pos", "ID", "ref", "alt", "QUAL", "FILTER",  "INFO"])
print(vcf)
vcf.pos -= 1
vcf.chromosome = "Chr" + vcf.chromosome.astype(str)
print(vcf)
vcf.to_parquet(output_path, index=False)