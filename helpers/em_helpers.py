# em.py helpers


def parse_samples(filepath):
    # Parses haplotype/allele samples for any number of individuals
    individuals = []
    z_init = []
    with open(filepath, 'r') as file:
        lines = file.readlines()
        num_i = int(lines[0].split(' ')[1])
        num_s = int(lines[1].split(' ')[1])
        samples = [line for line in lines[2:] if line != '\n']
        for i in range(num_i):
            individual = []
            z_indiv = []
            for s in range(num_s):
                index = i*num_s*2+s*2
                allele1 = samples[index].strip('\n')
                allele2 = samples[index+1].strip('\n')
                haplotype = (allele1, allele2)
                individual.append(haplotype)
                z_indiv.append(1/(num_s))
            individuals.append(individual)
            z_init.append(z_indiv)
        return individuals, z_init


def maximization(individuals, z_scores):
    # Returns new global allele frequencies
    global_freq = {}
    allele_count = 0
    for i in range(len(individuals)):
        for j in range(len(individuals[i])):
            haplotype = individuals[i][j]
            z = z_scores[i][j]
            for k in range(len(haplotype)):
                allele = haplotype[k]
                if allele not in global_freq:
                    global_freq[allele] = 0
                global_freq[allele] += 2*z
                allele_count += 1
    for key, value in global_freq.items():
        global_freq[key] = value / allele_count
    return global_freq


def expectation(individuals, freq):
    # Returns new z_scores for individual haplotyping
    z_scores = []
    has_converged = True
    for i in range(len(individuals)):
        z_indiv = []
        haplotypes = individuals[i]
        denom = sum([freq[hap[0]]*freq[hap[1]] for hap in haplotypes])
        for haplotype in haplotypes:
            h_prob = freq[haplotype[0]]*freq[haplotype[1]] / denom
            z_indiv.append(h_prob)
            test_z = h_prob
            has_converged &= ((test_z == 1.0) or (test_z == 0.0))
        z_scores.append(z_indiv)
    return z_scores, has_converged


def pretty_print(individuals, z_scores, global_freq):
    # Pretty-prints EM algorithm state
    print("Haplotypes")
    for i in range(len(individuals)):
        for j in range(len(individuals[i])):
            print(f"{individuals[i][j]}\tZ_{i+1}{j+1} = {z_scores[i][j]}")
    print("\nFrequencies")
    for key, val in global_freq.items():
        print(f"{key}\t{round(val, 4)}")
