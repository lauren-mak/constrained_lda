import click
from math import inf
import numpy as np
import pickle as pkl
from os.path import basename, join
from api import *


def prefix(filename):
    return '.'.join(basename(filename).split('.')[:-2]) # Cuts off the directory information and suffix


@click.group()
def main():
    pass


@main.command('model_application')
@click.argument('doc_word_file') 
@click.argument('topic_word_file') # *.raw_tw.csv
@click.argument('main_dir')
@click.argument('k', type = click.INT)
@click.argument('eta', type = click.FLOAT, default = 0.01)
def model_application(doc_word_file, topic_word_file, main_dir, k, eta):
    outdir = check_make(main_dir, 'output')
    dir_prefix = join(outdir, prefix(doc_word_file) + '.' + str(k) + '_' + str(eta)) # numvar.train-test.attempt.k_eta 
    test_doc_wrd = np.genfromtxt(doc_word_file, delimiter = ',')
    train_top_wrd = np.genfromtxt(topic_word_file, delimiter = ',')
    test_doc_top = np.linalg.solve(a, b)
    np.savetxt(dir_prefix + '.doc_topic.csv', best_gamma, fmt='%.5f', delimiter=',', newline='\n', header='')
    transformed_lambda = np.where(train_top_wrd < 0.01, 0, 1)
    np.savetxt(dir_prefix + '.topic_wrd.csv', transformed_lambda, fmt='%.5f', delimiter=',', newline='\n', header='')


@main.command('constrained_lda')
@click.argument('infile')
@click.argument('main_dir')
@click.argument('k', type = click.INT)
@click.argument('rounds', type = click.INT, default = 10)
@click.argument('iterations', type = click.INT, default = 100)
@click.argument('gp_iters', type = click.INT, default = 10)
@click.argument('gp_thresh', type = click.FLOAT, default = 0.001)
@click.argument('alpha', type = click.FLOAT, default = 0.1)
@click.argument('eta', type = click.FLOAT, default = 0.01)
@click.option('-c', '--constraints', type = click.File(), help='Constraint file')
@click.option('--debug', flag_value=True)
def constrained_lda(infile, main_dir, k, rounds, iterations, gp_iters, gp_thresh, alpha, eta, constraints, debug):

    # Make space for output files
    outdir = check_make(main_dir, 'output')
    dir_prefix = join(outdir, prefix(infile) + '.' + str(k) + '_' + str(eta)) # numvar.train-test.attempt.k_eta

    best_logl = inf
    best_logl_lst = []
    best_model = None
    best_gamma = None

    # Run LDA with different initial lambdas
    for i in range(rounds):
        logger(f'Round {i}: Initializing constrained LDA model', True)
        model = OnlineLDA(infile, constraints, k, alpha, eta, gp_iters, gp_thresh, debug)

        # Update estimates of gamma and lambda 
        logl_lst = []
        for j in range(iterations):
            if (j % 10) == 0: 
                logger(f'Iteration {j}', debug)
            (gamma, logl) = model.update_lambda()
            logl_lst += [logl]

        # Save this round's result if it has the best log-likelihood
        if logl < best_logl:
            best_logl = logl
            best_logl_lst = logl_lst
            best_model = model
            best_gamma = gamma
                
    # Output the optimal pickled model, document-topic frequency matrix, topic-word count matrix, and log-likelihoods. 
    # dir_prefix = # numvar.train-test.attempt.k_eta
    # numvar.train-test.attempt.k_eta.mtx-type.csv
    logger('Finished inferring strains and composition, printing results', True)
    np.savetxt(dir_prefix + '.doc_topic.csv', best_gamma, fmt='%.5f', delimiter=',', newline='\n', header='')
    np.savetxt(dir_prefix + '.raw_tw.csv', best_model._lambda, fmt='%.5f', delimiter=',', newline='\n', header='')
    transformed_lambda = np.where(best_model._lambda < 0.01, 0, 1)
    np.savetxt(dir_prefix + '.topic_wrd.csv', transformed_lambda, fmt='%.5f', delimiter=',', newline='\n', header='')
    np.savetxt(dir_prefix + '.logl.csv', best_logl_lst, fmt='%.5f', delimiter=',', newline='\n', header='')
    with open(dir_prefix + '.pkl','wb') as f:
        pkl.dump(best_model, f)


if __name__ == '__main__':
    main()
