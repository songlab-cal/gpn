# Whole-genome alignment processing

Code for processing WGA of the five non-human species. For the human multiz100way and cactus447way alignments, please refer to [GPN-MSA](https://www.nature.com/articles/s41587-024-02511-w) and [PhyloGPN](https://link.springer.com/chapter/10.1007/978-3-031-90252-9_7).
The pipeline should be generally applicable to other WGA in MAF format.

```bash
conda env create -f workflow/envs/general.yaml
conda activate wga_processing
snakemake --cores all
```
