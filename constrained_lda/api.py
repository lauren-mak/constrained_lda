import click
from datetime import datetime
import numpy as np
from os.path import basename, join
from scipy.special import gammaln, psi

def logger(i, d):
    if d: 
        print_string = f'{datetime.now().strftime("%H:%M:%S")}: {i}'
        click.echo(print_string, err=True)

def dirichlet_expectation(alpha):
    """
    For a vector theta ~ Dir(alpha), computes E[log(theta)] given alpha.
    """
    if (len(alpha.shape) == 1):
        return(psi(alpha) - psi(np.sum(alpha)))
    return(psi(alpha) - psi(np.sum(alpha, 1))[:, np.newaxis])

class OnlineLDA:
    """
    Implements online VB for LDA as described in (Hoffman et al. 2010).
    """

    def __init__(self, doc_word_file, num_topics, alpha, eta, kappa, gp_iters, gp_thresh, debug):
        """
        Arguments:
        alpha: Hyperparameter for prior on weight vectors theta (document-topic distribution)
        eta: Hyperparameter for prior on topics beta (topic-word distribution)
        kappa: Learning rate: exponential decay rate---should be between (0.5, 1.0] to guarantee asymptotic convergence.
        """
        self.doc_word_df = np.genfromtxt(doc_word_file, delimiter = ',') # np.2darray of self.num_docs x self.num_topics
        self.num_topics = int(num_topics)
        self.num_words = self.doc_word_df.shape[1]
        self.num_docs = self.doc_word_df.shape[0] 
        self._alpha = float(alpha)
        self._eta = float(eta)
        self._kappa = float(kappa)

        # Other runtime parameters:
        self.gp_iters = int(gp_iters)
        self.gp_thresh = float(gp_thresh)
        self.debug = debug

        # Initialize the variational distribution q(beta|lambda)
        self._lambda = 1*np.random.gamma(100., 1./100., (self.num_topics, self.num_words))
        self._Elogbeta = dirichlet_expectation(self._lambda)
        self._expElogbeta = np.exp(self._Elogbeta)

    def e_step(self):
        """
        E step: Calculates the expectation of gamma and phi given current lambda. These values will be used in the M-step to update the variational parameter lambda.  
        """
        # Initialize the variational distribution q(theta|gamma) 
        gamma = 1*np.random.gamma(100., 1./100., (self.num_docs, self.num_topics)) 
        Elogtheta = dirichlet_expectation(gamma) # np.2darray of self.num_docs x self.num_topics
        expElogtheta = np.exp(Elogtheta)

        sstats = np.zeros(self._lambda.shape)
        # For each document d update that document's gamma and phi
        for d in range(0, self.num_docs):
            logger('Document ' + str(d), self.debug)
            cts = self.doc_word_df[d,:] # np.1darray of length self.num_topics
            gammad = gamma[d, :] # np.1darray of length self.num_topics
            Elogthetad = Elogtheta[d, :] # np.1darray of length self.num_topics
            expElogthetad = expElogtheta[d, :] # np.1darray of length self.num_topics
            expElogbetad = self._expElogbeta # np.2darray of self.num_docs x self.num_topics
            # The optimal phi_{dwk} is proportional to  expElogthetad_k * expElogbetad_w. phinorm is the normalizer.
            phinorm = np.dot(expElogthetad, expElogbetad) + 1e-100 # Dot product i.e.: sum product over the last axis of expElogbetad and expElogthetad
            logger('Initial gamma ' + str(d) + ' ' + str(gammad), self.debug)
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
                logger(str(i) + ' Gamma ' + str(d) + ' ' + str(gammad), self.debug)
                logger(str(i) + ' deltaGamma ' + str(d) + ' ' + str(delta_gamma), self.debug)
                if (delta_gamma < self.gp_thresh):
                    break
            gamma[d, :] = gammad
            # Contribution of document d to the expected sufficientstatistics for the M step.
            sstats += np.outer(expElogthetad.T, cts/phinorm) # np.2darray of self.num_docs x self.num_topics

        # This step finishes computing the sufficient statistics for the M step, so that...
        # sstats[k, w] = \sum_d n_{dw} * phi_{dwk} = \sum_d n_{dw} * exp{Elogtheta_{dk} + Elogbeta_{kw}} / phinorm_{dw}.
        sstats = sstats * self._expElogbeta
        logger('Sstats ' + str(sstats), self.debug)

        return ((gamma, sstats))

    def m_step(self, sstats):
        """
        M step: Update values of lambda and Elogbeta based on the summary statistics. 
        """
        self._lambda = self._lambda + self._eta + self.num_docs * sstats / self.num_words
        self._Elogbeta = dirichlet_expectation(self._lambda)
        self._expElogbeta = np.exp(self._Elogbeta)
        logger('Lambda ' + str(self._lambda), self.debug)
        logger('Elogbeta ' + str(self._Elogbeta), self.debug)
        logger('expElogbeta ' + str(self._expElogbeta), self.debug)
        return 0
        # TODO Add calculation of constrain weights here (since after updating lambda)

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
        logger('E[log p(theta | alpha) - log q(theta | gamma)] = ' + str(doc_score), True)

        topic_score = 0
        # E[log p(beta | eta) - log q (beta | lambda)]
        topic_score = topic_score + np.sum((self._eta-self._lambda)*self._Elogbeta) 
        topic_score = topic_score + np.sum(gammaln(self._lambda) - gammaln(self._eta))
        topic_score = topic_score + np.sum(gammaln(self._eta*self.num_words) - gammaln(np.sum(self._lambda, 1)))
        logger('E[log p(beta | eta) - log q (beta | lambda)] = ' + str(topic_score), True)
        # TODO Add constraint weight term here 

        return(doc_score + topic_score)

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
        logger('Logl: ' + str(logl), True)

        return(gamma, logl)
