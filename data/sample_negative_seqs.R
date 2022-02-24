library("gkmSVM")
args = commandArgs(trailingOnly=TRUE)
print(args)

genome_name <- args[1]
library(genome_name, character.only=TRUE)
genome <- eval(parse(text=genome_name))
print(genome)

genNullSeqs(
  inputBedFN = args[2],
  outputBedFN = args[3],
  outputPosFastaFN = args[4],
  outputNegFastaFN = args[5],
  xfold = 1,
  repeat_match_tol = 0.02,
  GC_match_tol = 0.02,
  length_match_tol = 0.0,
  batchsize = 5000,
  nMaxTrials = 20,
  genome = genome)
