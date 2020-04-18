"""
Caleb Ellington
2020/03/15

Quantitiative Trait Loci analysis
"""
import argparse
import numpy as np
from os.path import isfile, isdir
from helpers.qtl_helpers import *

"""
Runs a quantitiative trait loci analysis for genotype/phenotpye data  
"""
def main():
    parser = argparse.ArgumentParser(description="quantitative trait loci analysis")

    parser.add_argument(
        "--genotype",
        "-g",
        action="store",
        default="data/genotype.txt",
        help="Tab separated genotype data",
    )

    parser.add_argument(
        "--phenotype",
        "-p",
        action="store",
        default="data/phenotype.txt",
        help="Tab separated phenotype data",
    )

    parser.add_argument(
        "--outdir",
        "-o",
        action="store",
        default="outputs/",
        help="Output directory"
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=False,
        help="Verbose run",
    )

    args = parser.parse_args()
    if not (isfile(args.genotype) and isfile(args.phenotype) and isdir(args.outdir)):
        print("Invalid filepath")
        exit(1)

    genotype = parse_genotype(args.genotype)
    phenotype = parse_phenotype(args.phenotype)
    qtl = QTL(phenotype, genotype, args.outdir)

    # Q1
    lods = qtl.get_lod_scores()
    # lod_threshold = qtl.get_lod_threshold(k=1000)
    # print(lod_threshold)
    lod_threshold = 10.525480649409918
    qtl_genes = np.argwhere(lods > lod_threshold)[:,0]
    print(qtl_genes)
    print(lods[qtl_genes])


    # Q2
    qtl.test_multi_marker_alpha()
    qtl.loocv_alphas()


if __name__ == "__main__":
    main()
