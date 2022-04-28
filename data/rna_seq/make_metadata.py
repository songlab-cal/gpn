import pandas as pd


# wget https://tools.1001genomes.org/api/accessions.csv?query=SELECT%20*%20FROM%20tg_accessions%20ORDER%20BY%20id -O accessions.csv
accessions = pd.read_csv("input/accessions.csv", header=None, usecols=[0, 1, 2, 3, 5, 6, 7, 9, 10,])
accessions.columns = ["accession_id",  "sequenced_by", "name", "country", "lat", "long", "collector", "cs_number", "admixture_group",]
print(accessions)
accessions.to_csv("metadata.tsv", sep="\t", index=False)
