import click
import matplotlib.pyplot as plt
import numpy as np
from os.path import basename, join
import pandas as pd
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import matthews_corrcoef
from seaborn import boxplot, stripplot

@click.group()
def main():
    pass

def calculate_mcc(actual_file, predicted_file, threshold):
    actual_alleles = np.genfromtxt(actual_file, delimiter = ',')
    predicted_alleles = np.genfromtxt(predicted_file, delimiter = ',')
    num_actual_strains = actual_alleles.shape[0]
    num_predicted_strains = predicted_alleles.shape[0]
    num_positions = actual_alleles.shape[1]

    mcc_mtx = np.zeros((num_actual_strains, num_predicted_strains))
    closest_mcc = []
    closest_to_actual = []
    num_reconstructed_strains = 0.0
    for i in range(num_actual_strains):
        for j in range(num_predicted_strains): 
            mcc_mtx[i][j] = matthews_corrcoef(actual_alleles[i], predicted_alleles[j])
        largest_mcc = np.amax(mcc_mtx[i])
        largest_mcc_idx = np.where(mcc_mtx[i] == largest_mcc)
        closest_mcc.append(largest_mcc)
        closest_to_actual.append(largest_mcc_idx[0][0])
        if 1 - np.linalg.norm(actual_alleles[i] - predicted_alleles[largest_mcc_idx], ord = 1) / num_positions  > threshold:
            num_reconstructed_strains += 1.0
    return np.mean(closest_mcc), closest_to_actual, num_reconstructed_strains / num_actual_strains

def calculate_jsd(actual_file, predicted_file, closest_to_actual):
    actual_freqs = np.genfromtxt(actual_file, delimiter = ',')
    predicted_freqs = np.genfromtxt(predicted_file, delimiter = ',')
    jsd = []
    num_pools = actual_freqs.shape[0]
    num_actual_strains = actual_freqs.shape[1]
    for i in range(num_pools):
        matched_strain_freqs = np.zeros(num_actual_strains)
        for j in range(num_actual_strains): 
            matched_strain_freqs[j] = predicted_freqs[i,closest_to_actual[j]]
        jsd.append(jensenshannon(actual_freqs[i], matched_strain_freqs))
    return np.mean(jsd)

def make_summary_stats(actual_prefix, predict_prefix, k, e, num_tests, threshold):
    mcc_lst = []
    crr_lst = []
    jsd_lst = []
    for t in range(num_tests):
        test_actual_prefix = actual_prefix + '.' + str(t)
        test_predict_prefix = predict_prefix + '.' + str(t) + '.' + k + '_' + e
        mcc, closest_to_actual, crr = calculate_mcc(test_actual_prefix + '.topic_wrd.csv', test_predict_prefix + '.topic_wrd.csv', threshold)
        mcc_lst.append(mcc)
        crr_lst.append(crr)
        jsd_lst.append(calculate_jsd(test_actual_prefix + '.doc_topic.csv', test_predict_prefix + '.doc_topic.csv', closest_to_actual))
    return mcc_lst, crr_lst, jsd_lst

def make_graph_and_table(arr, names, stat, results_prefix): 

    df_list = []
    means = []
    stdev = []
    for i, a in enumerate(arr):
        df_list.append(pd.DataFrame([[names[i]] * len(a), a]).transpose())
        means.append(np.mean(a))
        stdev.append(np.std(a))
    df = pd.concat(df_list)
    df.columns = ['Parameters (k, eta)', stat]

    plt.figure(figsize=(16,10), dpi= 120)
    boxplot(x = 'Parameters (k, eta)', y = stat, data = df, hue = 'Parameters (k, eta)')
    stripplot(x = 'Parameters (k, eta)', y = stat, data = df, color='black', size = 3, jitter = 1)
    plt.savefig(results_prefix + '.' + stat + '.png')

    return [means, stdev]


# File-name convention (actual): actual_dir/numvar_numstr.attempt.mtx-type.csv
# File-name convention (predicted): predict_dir/numvar_numstr.train-test.k_eta.attempt.mtx-type.csv
@main.command('eval_compositions')
@click.argument('actual_dir') 
@click.argument('predict_dir') 
@click.argument('results_dir') 
@click.argument('prefix') # numvar_numstr
@click.argument('k_values')
@click.argument('eta_values')
@click.argument('num_tests') # TODO set default at 50
@click.argument('threshold') # TODO set default at 90%
@click.argument('data_type') # train or test
def eval_compositions(actual_dir, predict_dir, results_dir, prefix, k_values, eta_values, num_tests, threshold, data_type):
    compare_mcc = []
    compare_crr = []
    compare_jsd = []
    k_list = k_values.split(',')
    eta_list = eta_values.split(',')
    param_names = []
    for k in k_list:
        for e in eta_list:
            param_names.append('k = ' + str(k) + ', eta = ' + str(e))
            actual_prefix = actual_dir + '/' + prefix + '.' + data_type
            predict_prefix = predict_dir + '/' + prefix + '.' + data_type
            mcc_lst, crr_lst, jsd_lst = make_summary_stats(actual_prefix, predict_prefix, k, e, int(num_tests), float(threshold)) 
            compare_mcc.append(mcc_lst)
            compare_crr.append(crr_lst)
            compare_jsd.append(jsd_lst)
    results_prefix = results_dir + '/' + prefix
    mcc_df = pd.DataFrame(make_graph_and_table(compare_mcc, param_names, 'Matthews_Corr_Coef', results_prefix), index = ['Mean_MCC', 'StDev_MCC'], columns = param_names).T
    crr_df = pd.DataFrame(make_graph_and_table(compare_crr, param_names, 'Correct_Recon_Rate', results_prefix), index = ['Mean_CRR', 'StDev_CRR'], columns = param_names).T
    jsd_df = pd.DataFrame(make_graph_and_table(compare_jsd, param_names, 'Jensen_Shannon_Div', results_prefix), index = ['Mean_JSD', 'StDev_JSD'], columns = param_names).T
    full_df = pd.concat([mcc_df, crr_df, jsd_df], axis=1)
    full_df.to_csv(results_prefix + '.summary.csv')
    # Calculate average MCC/CRR/JSD per k_eta
    #       avg_mcc std_mcc avg_crr avg_jsd
    # k_eta

if __name__ == '__main__':
    main()
