"""
Changes:
- do_e_step -> e_step
- Wrapped up update step in a function called m_step()
- approx_bound -> calc_likelihood
- Added self.doc_word_df to represent loaded data
- Converted meanchangethresh = 0.001 as a set parameter to self.gp_threshold
- Added self.gp_iters to expose the gamma-phi iteration parameter
- Deleted all usages of rhot and related values (ex. self._tau0, scaling for subsampling of the population of documents) because all documents used to update lambda and gamma
- Original: wordids = all_docs[doc[words]], wordcts = all_docs[doc[counts]]
- New: self.doc_word_df = np.2darray[docs,words -> cts]
- Original -> New: Got rid of ids = wordids[d]
- Original: cts = wordcts[d] 
- New: cts = self.doc_word_df[d,:]
- Original: self._expElogbeta[:, ids] which represents only the words in the document
- New: self._expElogbeta to represent all words in all documents
- Original -> New: Got rid of ids = wordids[d] 
- Original: cts = np.array(self.doc_word_df[d,:])
- New: cts = np.array(self.doc_word_df[d,:])
- Original -> New: len(ids) -> self.num_topics
- Original -> New: Removed self._updatect += 1
- Original -> Got rid of self.vocab, though may have to re-add if numpy doesn't allow naming arrays.
TODO:
- Figure out how I want to track gamma and logl
- Write an output function for document x topic and topic x word matrices
    - Something like n.savetxt('/tmp/lambda%d' % i, model._lambda.T)
- Write constraint-reading function prior to the initiation of EM
    - Recall that self._lambda = 1*n.random.gamma(100., 1./100., (self._K, self._W)) topics x words -> see which pairs are non-zero
- Instead of re-initializing gamma each time, use the previous value of gamma? 
- Loggers for gammad and update iterations
"""
import click
from configparser import ConfigParser
import pickle as pkl
from api import *


def prefix(filename):
    return '.'.join(basename(filename).split('.')[:-1]) # Cuts off the directory information and suffix


@click.group()
def main():
    pass


@main.command('constrained_lda')
@click.argument('cfg')
@click.option('--debug', flag_value=True)
def constrained_lda(cfg, debug):

    # Read in parameters from config file
    config = ConfigParser()
    config.read(cfg)
    default_params = config['default']
    hyper_params = config['hyper_params']
    runtime_params = config['runtime']

    # Initialize constrained LDA model
    logger('Initializing constrained LDA model', True)
    infile = default_params['input']
    model = OnlineLDA(infile, hyper_params['num_topics'], hyper_params['alpha'], hyper_params['eta'], hyper_params['kappa'], runtime_params['gp_iters'], runtime_params['gp_thresh'], debug)
    logger('Doc_word_df ' + str(model.doc_word_df), debug)
    logger('Lambda ' + str(model._lambda), debug)
    logger('Elogbeta ' + str(model._Elogbeta), debug)
    logger('expElogbeta ' + str(model._expElogbeta), debug)

    # Update estimates of gamma and lambda 
    logl_lst = []
    for i in range(int(runtime_params['update_iters'])):
        logger('Iteration ' + str(i), debug)
        (gamma, logl) = model.update_lambda()
        logl_lst += [logl]

    # Output the pickled model, document-topic frequency matrix, topic-word count matrix, and log-likelihoods at each EM iteration. 
    logger('Finished inferring strains and composition, printing results', True)
    dir_prefix = join(default_params['outdir'], prefix(infile))
    with open(dir_prefix + '.pkl','wb') as f:
        pkl.dump(model, f)
    np.savetxt(dir_prefix + '.doc_topic.csv', gamma, fmt='%.5f', delimiter=',', newline='\n', header='')
    np.savetxt(dir_prefix + '.topic_wrd.csv', model._lambda, fmt='%.5f', delimiter=',', newline='\n', header='')
    np.savetxt(dir_prefix + '.logl.csv', logl_lst, fmt='%.5f', delimiter=',', newline='\n', header='')


if __name__ == '__main__':
    main()
