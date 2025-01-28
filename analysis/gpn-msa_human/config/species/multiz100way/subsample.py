import numpy as np
import pandas as pd


n = 51
input_path = "89.txt"
output_path = f"vertebrates/{n}.txt"
force_keep = 3

rng = np.random.default_rng(seed=42)

input_species = pd.read_csv(input_path, header=None).values.ravel().tolist()
output_species = (
    input_species[:force_keep] +
    rng.choice(input_species[force_keep:], size=n-force_keep, replace=False).tolist()
)
# Sort the subsample to have the same order as in the original list
output_species = sorted(output_species, key=lambda x: input_species.index(x))

print(len(output_species))
print(output_species)

pd.DataFrame(output_species).to_csv(output_path, header=False, index=False)
