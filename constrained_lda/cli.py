import click
from configparser import ConfigParser
import pickle as pkl
from os import makedirs
from os.path import basename, join, exists
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
    model = OnlineLDA(infile, hyper_params['num_topics'], hyper_params['alpha'], hyper_params['eta'], runtime_params['gp_iters'], runtime_params['gp_thresh'], debug)
    logger(f'Doc_word_df {model.doc_word_df}', debug)
    logger(f'Lambda {model._lambda}', debug)
    logger(f'Elogbeta {model._Elogbeta}', debug)
    logger(f'expElogbeta {model._expElogbeta}', debug)

    # Update estimates of gamma and lambda 
    logl_lst = []
    for i in range(int(runtime_params['update_iters'])):
        logger(f'Iteration {i}', debug)
        (gamma, logl) = model.update_lambda()
        logl_lst += [logl]

    # Output the pickled model, document-topic frequency matrix, topic-word count matrix, and log-likelihoods at each EM iteration. 
    logger('Finished inferring strains and composition, printing results', True)
    outdir = default_params['outdir']
    if not exists(outdir):
        makedirs(outdir)
    dir_prefix = join(outdir, prefix(infile))
    with open(dir_prefix + '.pkl','wb') as f:
        pkl.dump(model, f)
    np.savetxt(dir_prefix + '.doc_topic.csv', gamma, fmt='%.5f', delimiter=',', newline='\n', header='')
    np.savetxt(dir_prefix + '.topic_wrd.csv', np.rint(model._lambda), fmt='%.5f', delimiter=',', newline='\n', header='')
    np.savetxt(dir_prefix + '.logl.csv', logl_lst, fmt='%.5f', delimiter=',', newline='\n', header='')


if __name__ == '__main__':
    main()
