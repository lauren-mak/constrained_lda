import click
from datetime import datetime
from math import factorial, inf
import numpy as np
from os.path import basename, dirname, join, exists
from os import makedirs
from scipy.special import gammaln, psi
from sklearn.preprocessing import normalize 

def logger(i, d):
    if d: 
        print_string = f'{datetime.now().strftime("%H:%M:%S")}: {i}'
        click.echo(print_string, err=True)

def check_make(directory, subdir):
    outdir = join(output, subdir)
    if not exists(outdir):
        makedirs(outdir)
    return outdir

def dirichlet_expectation(alpha):
    """
    For a vector theta ~ Dir(alpha), computes E[log(theta)] given alpha.
    """
    if (len(alpha.shape) == 1):
        return(psi(alpha) - psi(np.sum(alpha)))
    return(psi(alpha) - psi(np.sum(alpha, 1))[:, np.newaxis])

def binom(n, k):
    return factorial(n) // factorial(k) // factorial(n - k)

def biallelic_constraints(two_word_mtx):
    """
    Converts two-word topic-word matrix into a biallelic linkage matrix
    """
    linkage_mtx = np.zeros((3, 2))
    for row in two_word_mtx:
        first_allele = row[0] > 0
        secnd_allele = row[1] > 0 
        if first_allele and secnd_allele:
            linkage_mtx[0] += row
        elif first_allele > 0:
            linkage_mtx[1] += row
        elif secnd_allele > 0:
            linkage_mtx[2] += row
    linkage_mtx /= np.array([linkage_mtx.min(axis = 1)]).T # Scales each strain by the minimum allele weight in each row
    return linkage_mtx

def make_biallelic_list(df):
    """
    Converts full topic-word matrix into a list of biallelic linkage matrices
    """
    biallelic_list = np.zeros((binom(self.num_words, 2), 3, 2))
    for i in range(self.num_words):
        for j in range(i, self.num_words):
            biallelic_list[i + j - 1] = biallelic_constraints(df[:,(i,j)])
    return biallelic_list

def biallelic_distance(biallelic_lambda):
    """ 
    Sum of the Euclidean norms between the observed (constraints) and predicted (lambda) allelic pairs
    """
    score = 0
    for i in range(len(biallelic_list)):
        score += sum(np.linalg.norm(biallelic_lambda[i] - self.constraints[i], axis = 1))
    return 1./score if score != 0.0 else -inf

class OnlineLDA:
    """
    Implements online variational Bayes for LDA as described in (Hoffman et al. 2010).
    """

    def __init__(self, doc_word_file, constraint_file, num_topics, alpha, eta, gp_iters, gp_thresh, debug):
        """
        Arguments:
        alpha: Hyperparameter for prior on weight vectors theta (document-topic distribution)
        eta: Hyperparameter for prior on topics beta (topic-word distribution)
        kappa: Learning rate: exponential decay rate---should be between (0.5, 1.0] to guarantee asymptotic convergence.
        """
        self.doc_word_df = np.genfromtxt(doc_word_file, delimiter = ',') # np.2darray of self.num_docs x self.num_topics
        logger(f'Doc_word_df {self.doc_word_df}', debug)
        self.num_topics = int(num_topics)
        self.num_words = self.doc_word_df.shape[1]
        self.num_docs = self.doc_word_df.shape[0] 
        self._alpha = float(alpha)
        self._eta = float(eta)

        # Other runtime parameters:
        self.gp_iters = int(gp_iters)
        self.gp_thresh = float(gp_thresh)
        self.debug = debug

        # Initialize the variational distribution q(beta|lambda)
        self._lambda = 1*np.random.gamma(100., self._eta, (self.num_topics, self.num_words))
        self._Elogbeta = dirichlet_expectation(self._lambda)
        self._expElogbeta = np.exp(self._Elogbeta)
        logger(f'Lambda {self._lambda}', debug)
        logger(f'Elogbeta {self._Elogbeta}', debug)
        logger(f'expElogbeta {self._expElogbeta}', debug)

        # Store physical linkage information between variants.
        self.constraints = None
        if constraint_file:
            logger(f'Loading physical linkage information from {constraint_file}')
            constraint_mtx = np.genfromtxt(constraint_file, delimiter = ',')
            self.constraints = make_biallelic_list(constraint_mtx)

    def e_step(self):
        """
        E step: Calculates the expectation of gamma and phi given current lambda. These values will be used in the M-step to update the variational parameter lambda.  
        """
        # Initialize the variational distribution q(theta|gamma) 
        gamma = 1*np.random.gamma(100., 1./100., (self.num_docs, self.num_topics)) 
        Elogtheta = dirichlet_expectation(gamma) # np.2darray of self.num_docs x self.num_topics
        expElogtheta = np.exp(Elogtheta)

        sstats = np.zeros((self.num_topics, self.num_words))
        # For each document d update that document's gamma and phi
        for d in range(0, self.num_docs):
            # logger(f'Document {d}', self.debug)
            cts = self.doc_word_df[d,:] # np.1darray of length self.num_topics
            gammad = gamma[d, :] # np.1darray of length self.num_topics
            Elogthetad = Elogtheta[d, :] # np.1darray of length self.num_topics
            expElogthetad = expElogtheta[d, :] # np.1darray of length self.num_topics
            expElogbetad = self._expElogbeta # np.2darray of self.num_docs x self.num_topics
            # The optimal phi_{dwk} is proportional to  expElogthetad_k * expElogbetad_w. phinorm is the normalizer.
            phinorm = np.dot(expElogthetad, expElogbetad) + 1e-100 # Dot product i.e.: sum product over the last axis of expElogbetad and expElogthetad
            # logger(f'Initial gamma {d} {gammad}', self.debug)
            # Iterate between gamma and phi until convergence
            for i in range(0, self.gp_iters):
                lastgamma = gammad
                # We represent phi implicitly to save memory and time.
                # Substituting the value of the optimal phi back into the update for gamma gives this update. Cf. Lee&Seung 2001.
                gammad = self._alpha + expElogthetad * np.dot(cts / phinorm, expElogbetad.T)
                Elogthetad = dirichlet_expectation(gammad)
                expElogthetad = np.exp(Elogthetad)
                phinorm = np.dot(expElogthetad, expElogbetad) + 1e-100
                # Check threshold to see if gamma has converged. 
                delta_gamma = np.mean(abs(gammad - lastgamma))
                # logger(f'{i} Gamma {d} {gammad}', self.debug)
                # logger(f'{i} deltaGamma {d} {delta_gamma}', self.debug)
                if (delta_gamma < self.gp_thresh):
                    break
            gamma[d, :] = gammad
            # Contribution of document d to the expected sufficientstatistics for the M step.
            sstats += np.outer(expElogthetad.T, cts/phinorm) # np.2darray of self.num_docs x self.num_topics

        # This step finishes computing the sufficient statistics for the M step, so that...
        # sstats[k, w] = \sum_d n_{dw} * phi_{dwk} = \sum_d n_{dw} * exp{Elogtheta_{dk} + Elogbeta_{kw}} / phinorm_{dw}.
        sstats = sstats * self._expElogbeta
        logger(f'Sufficient Stats \n{sstats}', self.debug)
        gamma_row_norm = normalize(gamma, axis = 1, norm='l1')
        logger(f'Normalized Gamma \n{gamma_row_norm}', self.debug)

        return ((gamma_row_norm, sstats))

    def m_step(self, sstats):
        """
        M step: Update values of lambda and Elogbeta based on the summary statistics. 
        """
        logger(f'Lambda before maximization \n{self._lambda}', self.debug)
        self._lambda = self._lambda * (self._eta + self.num_docs * sstats / self.num_words)
        logger(f'Lambda after maximization \n{self._lambda}', self.debug)
        # self._lambda /= np.array([self._lambda.min(axis = 1)]).T # Scales each strain by the minimum variant weight in each row
        # logger(f'Lambda after scaling \n{self._lambda}', self.debug)
        self._Elogbeta = dirichlet_expectation(self._lambda)
        self._expElogbeta = np.exp(self._Elogbeta)
        logger(f'Elogbeta \n{self._Elogbeta}', self.debug)
        logger(f'expElogbeta \n{self._expElogbeta}', self.debug)
        return 0

    def calc_likelihood(self, gamma):
        """
        Estimate held-out likelihood (upper variational bound of poorly the current model explains the data) for new values of lambda. 
        gamma is the set of parameters to the variational distribution q(theta) corresponding to the set of documents passed in.
        """
        doc_score = 0
        Elogtheta = dirichlet_expectation(gamma)
        expElogtheta = np.exp(Elogtheta)
        # E[log p(docs | theta, beta)]
        for d in range(0, self.num_docs):
            cts = self.doc_word_df[d,:] # np.1darray of length self.self.num_words
            phinorm = np.zeros(self.num_words) # np.1darray of length self.num_words
            for i in range(0, self.num_words):
                temp = Elogtheta[d, :] + self._Elogbeta[:,d]
                tmax = max(temp)
                phinorm[i] = np.log(sum(np.exp(temp - tmax))) + tmax # Phi for each word in document d
            doc_score += np.sum(cts * phinorm) # Dot product? 
        # E[log p(theta | alpha) - log q(theta | gamma)]
        doc_score += np.sum((self._alpha - gamma) * Elogtheta)
        doc_score += np.sum(gammaln(gamma) - gammaln(self._alpha))
        doc_score += sum(gammaln(self._alpha*self.num_topics) - gammaln(np.sum(gamma, 1)))
        logger(f'E[log p(theta | alpha) - log q(theta | gamma)] = {doc_score}', True)

        # E[log p(beta | eta) - log q (beta | lambda)]
        topic_score = np.sum((self._eta-self._lambda)*self._Elogbeta) 
        topic_score += np.sum(gammaln(self._lambda) - gammaln(self._eta))
        topic_score += np.sum(gammaln(self._eta*self.num_words) - gammaln(np.sum(self._lambda, 1)))
        logger(f'E[log p(beta | eta) - log q (beta | lambda)] = {topic_score}', True)
        total_score = (doc_score + topic_score) 
        if self.constraints:
            constraint_weight = biallelic_distance(make_biallelic_list(self._lambda))
            logger(f'The predicted topic-word constraint weight is {constraint_weight}')
            total_score *= constraint_weight

        return total_score

    def update_lambda(self):
        """
        First, E step on the corpus to calculate gamma. Then uses that result to maximize (M step) the variational parameter matrix lambda.
        Returns: 
        Lambda: Estimate of the variational bound for the entire corpus for the OLD setting of lambda based on the documents passed in. IOW, topic x word counts (that are later converted to frequencies using printtopics.py). 
        Gamma: Parameters to the variational distribution over the topic weights theta for the documents analyzed in this update. IOW, document x topic frequencies.
        """
        # E step to update gamma, phi | lambda.
        (gamma, sstats) = self.e_step()
        # M step to update lambda.
        constraint_wts = self.m_step(sstats)
        # Estimate held-out likelihood for current values of lambda.
        logl = self.calc_likelihood(gamma)
        logger(f'Logl: {logl}', True)

        return(gamma, logl)
