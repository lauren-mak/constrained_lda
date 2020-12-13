import click
from configparser import ConfigParser
import numpy as np
import pickle as pkl
from os import makedirs
from os.path import basename, join, exists
from api import *
from support.eval_compositions import calc_mcc


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

    # Make space for output files
    outdir = default_params['outdir']
    if not exists(outdir):
        makedirs(outdir)
    infile = default_params['input']
    dir_prefix = join(outdir, prefix(infile))

    # Initialize constrained LDA model
    with open(dir_prefix + '.multi_init.csv', 'w') as results:
        for i in range(int(runtime_params['init_rounds'])):
            logger(f'Round {i}: Initializing constrained LDA model', True)
            model = OnlineLDA(infile, default_params['constraints'], hyper_params['num_topics'], hyper_params['alpha'], hyper_params['eta'], runtime_params['gp_iters'], runtime_params['gp_thresh'], debug)

            # Update estimates of gamma and lambda 
            logl_lst = []
            for j in range(int(runtime_params['update_iters'])):
                logger(f'Iteration {j}', debug)
                (gamma, logl) = model.update_lambda()
                logl_lst += [logl]
            with open(dir_prefix + '.pkl','wb') as f:
                pkl.dump(model, f)
                
            # Output the round, final log-likelihood, and lambda matrix. 
            results.write(f'Round {i}\n')
            results.write(f'Final log-likelihood: {logl_lst[-1]}\n')
            # transformed_lambda = np.where(model._lambda < 2.0, 0, 1)
            transformed_lambda = np.where(model._lambda < 0.01, 0, 1)
            results.write(f'Lambda after transforming \n{transformed_lambda}\n')
            mcc, strains, crr = calc_mcc(model.doc_word_df.astype(int), transformed_lambda, 0.9)
            results.write(f'Correct reconstruction rate: {crr}\n')

    # Output the optimal pickled model, document-topic frequency matrix, topic-word count matrix, and log-likelihoods. 
    # logger('Finished inferring strains and composition, printing results', True)
    # np.savetxt(dir_prefix + '.doc_topic.csv', gamma, fmt='%.5f', delimiter=',', newline='\n', header='')
    # np.savetxt(dir_prefix + '.topic_wrd.csv', model._lambda, fmt='%.5f', delimiter=',', newline='\n', header='')
    # np.savetxt(dir_prefix + '.logl.csv', logl_lst, fmt='%.5f', delimiter=',', newline='\n', header='')


if __name__ == '__main__':
    main()
