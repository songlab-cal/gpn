import pandas as pd
import re

# plantregmap.gao-lab.org/download.php
with open("plant_reg_map_source.txt", "r") as f:
    document = f.read()

# example: "download/Arabidopsis_thaliana/TFBS/TFBS_AT1G77450_Ath_seedling_normal.bed"
track_urls = re.findall(r"download/Arabidopsis_thaliana/\w*/[^/]*.bed", document)
track_urls = ["http://plantregmap.gao-lab.org/download_ftp.php?filepath=08-" + track_url for track_url in track_urls]
track_names = [track_url.split("/")[-1].split(".")[0] for track_url in track_urls]
tracks = pd.DataFrame(dict(track_name=track_names, track_url=track_urls)).set_index("track_name")
print(tracks)
tracks.to_csv("tracks.tsv", "\t")
