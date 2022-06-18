import pandas as pd


all_coordinates = [
    "Chr5:3564493-3565087",
    #"Chr5:3687412-3688325",
]


for coordinates in all_coordinates:
    print(coordinates)
    chromosome = coordinates.split(":")[0]
    start = int(coordinates.split(":")[1].split("-")[0])
    end = int(coordinates.split(":")[1].split("-")[1])
    df = pd.DataFrame(dict(pos=range(start, end)))
    df["chromosome"] = chromosome
    print(df)
    df.to_parquet(f"examples_{coordinates}.parquet", index=False)