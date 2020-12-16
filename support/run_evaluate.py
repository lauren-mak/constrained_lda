import csv
import luigi
from os.path import basename, dirname, join, exists
from os import makedirs
import pandas as pd
from random import randint, shuffle
import subprocess
import time

# PYTHONPATH='.' luigi --module run_evaluate LDA_For_Strains --workers 10

def check_make(directory, subdir):
    outdir = join(directory, subdir)
    if not exists(outdir):
        makedirs(outdir)
    return outdir


# sim_datasets/input/numvar.train-test.attempt.doc_word.csv
class Single_Solve_Usage(luigi.Task):
    num_variants = luigi.Parameter()
    master_dir = luigi.Parameter()
    k = luigi.Parameter()
    eta = luigi.Parameter()
    idx = luigi.IntParameter()

    def __init__(self, *args, **kwargs):
        super(Single_Solve_Usage, self).__init__(*args, **kwargs)
        self.input_dir = check_make(self.master_dir, 'input')
        self.output_dir = check_make(self.master_dir, 'output')
        # self.test_prefix = join(self.input_dir, self.num_variants + '.test.' + str(self.idx))

    def requires(self):
        train_input_prefix = join(self.input_dir, self.num_variants + '.train.' + str(self.idx))
        cmd = ['python','/Users/laurenmak/Dropbox/workspace/constrained_lda/constrained_lda/cli.py']
        subprocess.run(['python','/Users/laurenmak/Dropbox/workspace/constrained_lda/constrained_lda/cli.py', 'constrained_lda', train_input_prefix + '.doc_word.csv', self.output_dir, self.k, self.eta])
        # train_output_prefix = join(self.output_dir, self.num_variants + '.train.' + str(self.idx) + '.' + self.k + '_' + self.eta)
        # subprocess.run(cmd + ['lda_application', self.test_prefix + '.doc_word.csv', train_output_prefix + '.raw_tw.csv', self.output_dir])

    def output(self):
        return luigi.LocalTarget(join(self.output_dir, self.num_variants + '.train.' + str(self.idx) + '.' + self.k + '_' + self.eta) + '.topic_wrd.csv') 
        # join(self.output_dir, self.test_prefix + '.' + self.k + '_' + self.eta + '.topic_wrd.csv'))


class LDA_For_Strains(luigi.Task):
    attempts = luigi.IntParameter()
    num_variants = luigi.Parameter()
    master_dir = luigi.Parameter()
    k_values = luigi.Parameter()
    eta_values = luigi.Parameter()

    def __init__(self, *args, **kwargs):
        super(LDA_For_Strains, self).__init__(*args, **kwargs)
        self.gs_dir = check_make(self.master_dir, 'gold_standard')
        self.input_dir = check_make(self.master_dir, 'input')
        self.output_dir = check_make(self.master_dir, 'output')
        self.results_dir = check_make(self.master_dir, 'results')

    def requires(self):  
        info = []
        k_lst = self.k_values.split(',')
        e_lst = self.eta_values.split(',')
        for i in k_lst:
            for j in e_lst:
                for l in range(self.attempts):
                    info.append(Single_Solve_Usage(k = i, eta = j, idx = l, num_variants = self.num_variants, master_dir = self.master_dir))
        return info

    def output(self):
        return luigi.LocalTarget(join(self.results_dir, self.num_variants + '.summary.csv'))

    def run(self):
        cmd = ['python', '/Users/laurenmak/Dropbox/workspace/constrained_lda/support/eval_compositions.py', 'eval_compositions', self.gs_dir, self.output_dir, self.results_dir, self.num_variants, self.k_values, self.eta_values, str(self.attempts), '0.9', 'train']
        subprocess.run(cmd)
        # subprocess.run(cmd.append('test'))


if __name__ == '__main__':
    luigi.run(main_cls_task=LDA_For_Strains)
