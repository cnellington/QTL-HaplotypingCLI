"""
qtl.py helpers
"""


from math import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import LeaveOneOut


def parse_genotype(filepath):
    # Parses genotype data samples
    genotype = []
    with open(filepath, 'r') as file:
        data = file.readlines()
        for line in data:
            vals = line.split('\t')
            genotype.append([int(val) for val in vals])
    return np.array(genotype)


def parse_phenotype(filepath):
    # Parses phenotype data samples
    phenotype = []
    with open(filepath, 'r') as file:
        data = file.read().strip('\n').split('\t')
        phenotype = [float(val) for val in data]
    return np.array(phenotype)


class QTL:
    """
    Tools for LOD and linear regression analysis of quantitative
    trait loci analysis.
    """
    def __init__(self, phenotype, genotype, outdir):
        self.phenotype = phenotype
        self.genotype = genotype
        self.outdir = outdir

    def get_lod_scores(self):
        # Gets LOD scores of all genetic markers, saves a histogram to output
        lods = self._lod_score_helper(self.phenotype, self.genotype)
        plt.hist(lods, bins=20)
        plt.title("Marker LOD Histogram")
        plt.xlabel("LOD")
        plt.ylabel("Marker Count")
        plt.savefig(self.outdir+"marker_hist.png")
        return lods

    def get_lod_threshold(self, k=1000, null_threshold=0.95):
        # Gets the LOD significance threshold based on k samples
        # and the null hypothesis threshold.
        max_lods = []
        for i in range(k):
            if i % 10 == 0:
                print(f"{round(i/k, 2)*100}%")
            phenotype = np.random.permutation(self.phenotype)
            lods = self._lod_score_helper(phenotype, self.genotype)
            max_lods.append(np.amax(lods))
        max_lods.sort()
        threshold_index = int(null_threshold * len(max_lods))
        n,x,_ = plt.hist(max_lods)
        bin_centers = 0.5*(x[1:]+x[:-1])
        plt.clf()
        plt.plot(bin_centers, n)
        plt.title("No-QTL Maximum LOD Distribution")
        plt.xlabel("LOD")
        plt.savefig(self.outdir+"lod_noqtl_hist.png")
        return max_lods[threshold_index]

    def _lod_score_helper(self, phenotype, genotype):
        # Calculates all LOD scores for this phenotype/genotype combination
        lods = []
        mu_nq, sig_nq = self._get_pdf(phenotype)
        for gene_id in range(len(genotype)):
            expression = genotype[gene_id]
            on_ids = np.argwhere(expression == 1)
            off_ids = np.argwhere(expression != 1)
            mu_q_on, sig_q_on = self._get_pdf(phenotype[on_ids])
            mu_q_off, sig_q_off = self._get_pdf(phenotype[off_ids])

            qtl_prob = 1.0
            nq_prob = 1.0
            for pheno_id in range(len(phenotype)):
                pheno_val = phenotype[pheno_id]
                gene_val = expression[pheno_id]
                if gene_val == 1:
                    qtl_prob *= self._pnormal(pheno_val, mu_q_on, sig_q_on)
                else:
                    qtl_prob *= self._pnormal(pheno_val, mu_q_off, sig_q_off)
                nq_prob *= self._pnormal(pheno_val, mu_nq, sig_nq)
            lod = log10(qtl_prob/nq_prob)
            lods.append(lod)
        return np.array(lods)

    def _get_pdf(self, phenotype):
        # Returns (mu, sigma) for pdf
        return np.mean(phenotype), np.std(phenotype)

    def _pnormal(self, val, mu, sig):
        # Gets the probability of being at point val in a pdf
        # with mu meand and sig deviation
        return (1/(sig*sqrt(2*pi)))*exp(-0.5*((val-mu)/sig)**2)

    def test_multi_marker_alpha(self):
        # Tests alpha ranges for L1 and L2 regularization with linear regression
        # Saves the valid ranges as a MSE vs Regularization graph for estimating
        # best regularization parameters
        normal_genotype = []
        for genotype in self.genotype.T:
            normal_genotype.append(self._normalize(genotype))
        normal_genotype = np.array(normal_genotype)
        x_train = normal_genotype[:250]
        x_test = normal_genotype[250:]
        y_train = self.phenotype[:250]
        y_test = self.phenotype[250:]
        alpha1_range = np.linspace(0.08, 0.135, 500)
        alpha2_range = np.linspace(560.0, 3500.0, 500)

        l1_mse = []
        l2_mse = []
        for alpha in alpha1_range:
            mse1 = self._get_regression_mse(x_train,
                                            x_test,
                                            y_train,
                                            y_test,
                                            loss="L1",
                                            alpha=alpha)
            l1_mse.append(mse1)
        for alpha in alpha2_range:
            mse2 = self._get_regression_mse(x_train,
                                            x_test,
                                            y_train,
                                            y_test,
                                            loss="L2",
                                            alpha=alpha)

            l2_mse.append(mse2)
        plt.plot(alpha1_range, l1_mse)
        plt.title("L1 Linear Regression MSE")
        plt.xlabel("Regularization (C)")
        plt.ylabel("Test-set MSE")
        plt.savefig(self.outdir+"l1_mse.png")
        plt.clf()
        plt.plot(alpha2_range, l2_mse)
        plt.title("L2 Linear Regression MSE")
        plt.xlabel("Regularization (C)")
        plt.ylabel("Test-set MSE")
        plt.savefig(self.outdir + "l2_mse.png")

    def loocv_alphas(self):
        # Uses leave-one-out cross validation to determine the absolute
        # best regularization parameter for L1 and L2 regularized regression
        # within a range estimated from test_multi_marker_alpha
        normal_genotype = []
        for genotype in self.genotype.T:
            normal_genotype.append(self._normalize(genotype))
        normal_genotype = np.array(normal_genotype)
        normal_genotype
        alpha1_range = np.linspace(0.08, 0.135, 100)
        alpha2_range = np.linspace(3100.0, 3200.0, 100)
        loo = LeaveOneOut()
        alpha1_mse_min = 100.0
        alpha1_best = 0.0
        alpha2_mse_min = 100.0
        alpha2_best = 0.0
        for i in range(len(alpha1_range)):
            print(i)
            alpha1 = alpha1_range[i]
            alpha2 = alpha2_range[i]
            mse1_sum = 0.0
            mse2_sum = 0.0
            for train_index, test_index in loo.split(normal_genotype):
                x_train = normal_genotype[train_index]
                x_test = normal_genotype[test_index]
                y_train = self.phenotype[train_index]
                y_test = self.phenotype[test_index]
                mse1_sum += self._get_regression_mse(x_train, x_test,
                                                     y_train, y_test,
                                                     loss="L1", alpha=alpha1)
                mse2_sum += self._get_regression_mse(x_train, x_test,
                                                     y_train, y_test,
                                                     loss="L2", alpha=alpha2)
            mse1 = mse1_sum / len(normal_genotype)
            mse2 = mse2_sum / len(normal_genotype)
            if mse1 < alpha1_mse_min:
                alpha1_mse_min = mse1
                alpha1_best = alpha1
            if mse2 < alpha2_mse_min:
                alpha2_mse_min = mse2
                alpha2_best = alpha2
        print(f"Best L1 Alpha: {alpha1_best}, mse {alpha1_mse_min}")
        print(f"Best L2 Alpha: {alpha2_best}, mse {alpha2_mse_min}")

    def _get_regression_mse(self, x_train, x_test, y_train, y_test, loss="L1", alpha=0.1):
        # Runs and tests a single linear regression based on
        # train/test/loss/regularization(alpha) parameters
        if loss == "L1":
            regr = linear_model.Lasso(alpha=alpha)
        elif loss == "L2":
            regr = linear_model.Ridge(alpha=alpha)
        regr.fit(x_train, y_train)
        pred = regr.predict(x_test)
        return mean_squared_error(y_test, pred)

    def _normalize(self, arr):
        # Normalizes a distribution to have mean 0 and deviation 1
        mu, sig = self._get_pdf(arr)
        if sig == 0.0:
            return arr
        return (arr-mu)/sig


