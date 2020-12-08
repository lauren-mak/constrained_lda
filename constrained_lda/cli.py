import click
from configparser import ConfigParser
import pickle as pkl
from os.path import basename, join
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
