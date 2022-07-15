import gget
import pandas as pd


df = pd.read_csv("Species.csv")  # https://plants.ensembl.org/species.html
print(df)
df.drop_duplicates("Taxon ID", inplace=True)  # To remove multiple strains of the same species e.g. in wheat
print(df)
df = df.query('Classification == "eudicotyledons"')
print(df)

brassicales = pd.read_csv("genome_list.tsv", header=None).replace({"Arabidopsis_thaliana_train": "Arabidopsis_thaliana"}).values.astype(str).ravel()
print(brassicales)

df["species"] = df.Name.str.replace(" ", "_")
df = df[df.species.isin(brassicales)]
print(df)

df["fasta_url"] = df.species.apply(lambda s: gget.ref(s, which="dna", ftp=True)[0].replace(".dna.", ".dna_sm."))
print(df)

df.to_csv("species_metadata.tsv", sep="\t", index=False)
