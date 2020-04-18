"""
Caleb Ellington
CSE 427: Computational Biology
2020/03/17

EM-based haplotype reconstruction
"""
import argparse
from os.path import isfile
from helpers.em_helpers import *

"""
Runs an EM-based haplotype reconstruction
"""
def main():
    parser = argparse.ArgumentParser(description="haplotype reconstruction")

    parser.add_argument(
        "--samples",
        "-s",
        action="store",
        default="data/h_data.txt",
        help="Possible inidividual haplotype data",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=False,
        help="Verbose run",
    )

    args = parser.parse_args()
    if not isfile(args.samples):
        print("Invalid filepath")
        exit(1)

    individuals, z_scores = parse_samples(args.samples)
    global_freq = None
    has_converged = False
    while not has_converged:
        global_freq = maximization(individuals, z_scores)
        # pretty_print(individuals, z_scores, global_freq)
        z_scores, has_converged = expectation(individuals, global_freq)
        # pretty_print(individuals, z_scores, global_freq)
    pretty_print(individuals, z_scores, global_freq)


if __name__ == "__main__":
    main()
