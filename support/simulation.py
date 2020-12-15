import csv
import luigi
from os.path import basename, dirname, join, exists
from os import makedirs
import pandas as pd
from random import randint, shuffle
import subprocess
import time


# PYTHONPATH='.' luigi --module simulation Generate_Strain_Datasets --workers 10


def check_make(directory, subdir):
    outdir = join(directory, subdir)
    if not exists(outdir):
        makedirs(outdir)
    return outdir


def make_dict_from_list(lst):
    dct = dict()
    for i in lst:
      dct[i] = dct.get(i, 0) + 1
    return dct


# numvar.train-test.k_eta.attempt.mtx-type.csv
def make_inputs(unique_strains, counts, strains_per_sample, prefix, gs_dir, input_dir):
    all_strains = []
    for i, s in enumerate(unique_strains):
        if counts[i] > 0:
            all_strains += [s] * int(counts[i])
    shuffle(all_strains)
    split_strains = [all_strains[i:i + strains_per_sample] for i in range(0, len(all_strains), strains_per_sample)]
    pools_to_strains_lst = []
    for i, l in enumerate(split_strains):
        pool_dict = make_dict_from_list(l)
        pool_df = pd.DataFrame(pool_dict, index = [i], columns = pool_dict.keys()) 
        pools_to_strains_lst.append(pool_df)
    doc_topic_ct_mtx = pd.concat(pools_to_strains_lst, sort=False).fillna(0)
    doc_topic_mtx = doc_topic_ct_mtx.div(doc_topic_ct_mtx.sum(axis=1), axis=0) # gs_dir
    pool_strains = list(doc_topic_mtx.columns.values)
    strains_to_alleles_lst = []
    for s in pool_strains:
        strains_to_alleles_lst.append([int(char) for char in s])
    topic_wrd_mtx = pd.DataFrame(strains_to_alleles_lst, index = pool_strains) # gs_dir
    doc_word_mtx = doc_topic_ct_mtx.dot(topic_wrd_mtx) # input_dir
    doc_topic_mtx.to_csv(join(gs_dir, prefix + '.doc_topic.csv'), index = False, header = False)
    topic_wrd_mtx.to_csv(join(gs_dir, prefix + '.topic_wrd.csv'), index = False, header = False)
    doc_word_mtx.astype(int).to_csv(join(input_dir, prefix + '.doc_word.csv'), index = False, header = False)


def split_train_test(ms_out, test_ratio, num_variants):
    # Make lists of the unique strains and how many in total
    with open(ms_out, 'r') as m:
        all_strains = [line.rstrip() for line in m]
    strain_ct_dict = make_dict_from_list(all_strains[7:]) # Skip header
    zero_strain = '0' * num_variants
    if zero_strain in strain_ct_dict:
        print('Correcting a strain completely comprised of 0s')
        zero_alleles = ['0'] * num_variants
        zero_alleles[randint(0, num_variants - 1)] = '1'
        new_zero_strain = ''.join(zero_alleles)
        strain_ct_dict[new_zero_strain] = strain_ct_dict.get(new_zero_strain, 0) + strain_ct_dict[zero_strain]
        del strain_ct_dict[zero_strain]
    unique_strains = list(strain_ct_dict.keys())
    total_counts = list(strain_ct_dict.values())

    # Partition total set of strains into training and test sets
    test_counts = [round(x * test_ratio) for x in total_counts]
    train_counts = []
    for i in range(len(unique_strains)):
        train_counts.append(total_counts[i] - test_counts[i])
    num_test_strains = len(all_strains[7:]) * test_ratio
    difference = int(num_test_strains) - sum(test_counts)
    if difference > 0:
        max_idx = train_counts.index(max(train_counts))
        train_counts[max_idx] -= difference
        test_counts[max_idx] += difference
    elif difference < 0: 
        max_idx = test_counts.index(max(test_counts))
        test_counts[max_idx] -= difference
        train_counts[max_idx] += difference
    return unique_strains, train_counts, test_counts


class Make_Single_Dataset(luigi.Task):
    idx = luigi.Parameter()
    num_variants = luigi.Parameter()
    num_train_samples = luigi.IntParameter()
    num_test_samples = luigi.IntParameter()
    num_strains_per_sample = luigi.IntParameter()
    master_dir = luigi.Parameter()

    def __init__(self, *args, **kwargs):
        super(Make_Single_Dataset, self).__init__(*args, **kwargs)
        self.gs_dir = check_make(self.master_dir, 'gold_standard')
        self.input_dir = check_make(self.master_dir, 'input')
        self.info_dir = check_make(self.master_dir, 'info')
        self.prefix = self.num_variants + '.' + self.idx # numvar.attempt.mtx-type.csv

    def output(self):
        return luigi.LocalTarget(join(self.info_dir, self.num_variants + '.' + self.idx + '.csv'))

    def run(self):
        ms_out = join(self.gs_dir, self.num_variants + '.' + self.idx + '.txt')
        num_total_datasets = self.num_train_samples + self.num_test_samples
        num_total_strains = num_total_datasets * self.num_strains_per_sample
        with open(ms_out, 'w') as mout: 
            subprocess.Popen(['/Users/laurenmak/Programs/msdir/ms', str(num_total_strains), '1', '-s', self.num_variants, '-t', '0.000001'], stdout=mout)
        test_ratio = float(self.num_test_samples) / num_total_datasets
        time.sleep(0.5) # Needed so that ms output is saved before it is opened again
        unique_strains, train_counts, test_counts = split_train_test(ms_out, test_ratio, int(self.num_variants))
        make_inputs(unique_strains, train_counts, self.num_strains_per_sample, self.num_variants + '.train.' + self.idx, self.gs_dir, self.input_dir)
        make_inputs(unique_strains, test_counts, self.num_strains_per_sample, self.num_variants + '.test.' + self.idx, self.gs_dir, self.input_dir)
        with self.output().open('w') as out_csv:
            out_csv.write(f'{self.num_variants},{self.idx},{len(unique_strains)}')


class Generate_Strain_Datasets(luigi.Task):
    attempts = luigi.IntParameter()
    num_variants = luigi.Parameter()
    master_dir = luigi.Parameter()

    def __init__(self, *args, **kwargs):
        super(Generate_Strain_Datasets, self).__init__(*args, **kwargs)
        self.gs_dir = check_make(self.master_dir, 'gold_standard')
        self.info_dir = check_make(self.master_dir, 'info')

    def requires(self):  
        info = []  
        for i in range(self.attempts):
            info.append(Make_Single_Dataset(idx = str(i)))
        return info

    def output(self):
        return luigi.LocalTarget(join(self.info_dir, self.num_variants + '.summary.csv'))

    def run(self):
        with self.output().open('w') as out_list:
            out_list.write('Num_Vars,Iteration,Num_Unique_Strains\n')
            for info in self.input():
                with open(info.path, 'r') as f:
                    l = f.readlines()
                    out_list.write(l[0] + '\n')


if __name__ == '__main__':
    luigi.run(main_cls_task=Generate_Strain_Datasets)
