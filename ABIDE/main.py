import os
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn import preprocessing
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.feature_selection import RFE
from nilearn import datasets
import shutil
import csv
import glob
from nilearn import connectome
import scipy.io as sio
from scipy.spatial import distance
import warnings
warnings.filterwarnings("ignore")


def get_networks(data_folder, subject_list, kind, atlas_name="aal", variable='connectivity'):

    all_networks = []
    for subject in subject_list:
        fl = os.path.join(data_folder, subject,
                          subject + "_" + atlas_name + "_" + kind + ".mat")
        matrix = sio.loadmat(fl)[variable]
        all_networks.append(matrix)

    idx = np.triu_indices_from(all_networks[0], 1)
    norm_networks = [np.arctanh(mat) for mat in all_networks]
    vec_networks = [mat[idx] for mat in norm_networks]
    matrix = np.vstack(vec_networks)

    return matrix


def get_ids(subject_IDs_path, num_subjects=None):

    subject_IDs = np.genfromtxt(subject_IDs_path, dtype=str)

    if num_subjects is not None:
        subject_IDs = subject_IDs[:num_subjects]
    return subject_IDs


def get_subject_score(phenotype, subject_list, score):

    scores_dict = {}

    with open(phenotype) as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if row['SUB_ID'] in subject_list:
                scores_dict[row['SUB_ID']] = row[score]

    return scores_dict


def get_filename(order_list, folder_path, type_end):

    filenames = []
    os.chdir(folder_path)
    for ID_ in order_list:
        try:
            filenames.append(glob.glob('*' + ID_ + type_end)[0])
        except IndexError:
            filenames.append('N/A')

    return filenames


def subject_connectivity(subject_ID_, timeseries, atlas_name, kind):


    if kind in ['tangent', 'partial correlation', 'correlation']:
        conn_measure = connectome.ConnectivityMeasure(kind=kind)
        connectivity = conn_measure.fit_transform([timeseries])[0]    
        file_name_mat = subject_ID_ + '_' + atlas_name + '_' + kind.replace(' ', '_') + '.mat'

    return connectivity, file_name_mat 

def read_timeseries_transfromer_mat(folder_path, save_folder_path, subject_IDs, filenames_list, atlas_name, kind):

    data_mat_list = []
    for file_name_, subject_ID_ in zip(filenames_list, subject_IDs):
        ROIs_file_path = os.path.join(folder_path, file_name_)
        timeseries_ = np.loadtxt(ROIs_file_path, skiprows=0)
        connectivity, file_name_mat = subject_connectivity(subject_ID_, timeseries_, atlas_name, kind)
        save_dir = os.path.join(save_folder_path, subject_ID_)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        shutil.copyfile(ROIs_file_path, os.path.join(save_dir, file_name_))
        sio.savemat(os.path.join(save_dir, file_name_mat),{'connectivity': connectivity})
        data_mat_list.append(connectivity)
    return data_mat_list


def feature_selection(matrix, labels, train_ind, fnum):


    estimator = RidgeClassifier()
    selector = RFE(estimator, n_features_to_select=fnum, step=100, verbose=1)

    featureX = matrix[train_ind, :]
    featureY = labels[train_ind]
    selector = selector.fit(featureX, featureY.ravel())
    x_data = selector.transform(matrix)

    print("Number of labeled samples %d" % len(train_ind))
    print("Number of features selected %d" % x_data.shape[1])

    return x_data


def create_affinity_graph_from_scores(scores, subject_list, phenotype):

    num_nodes = len(subject_list)
    graph = np.zeros((num_nodes, num_nodes))

    for l in scores:
        label_dict = get_subject_score(phenotype, subject_list, l)
        # 量化表型分数
        if l in ['AGE_AT_SCAN', 'FIQ']:
            for k in range(num_nodes):
                for j in range(k + 1, num_nodes):
                    try:
                        val = abs(float(label_dict[subject_list[k]]) - float(label_dict[subject_list[j]]))
                        if val < 2:
                            graph[k, j] += 1
                            graph[j, k] += 1
                    except ValueError:  # missing label
                        pass

        else:
            for k in range(num_nodes):
                for j in range(k + 1, num_nodes):
                    if label_dict[subject_list[k]] == label_dict[subject_list[j]]:
                        graph[k, j] += 1
                        graph[j, k] += 1

    return graph


def parse_args():
    parser = argparse.ArgumentParser(description='ABIDE data processing script')
    parser.add_argument('--popgraph', action='store_true', help='Enable feature selection')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    pipeline = 'cpac'
    root_folder = os.getcwd()

    files = ['rois_ho']
    num_subjects = 871
    filemapping = {'func_preproc': '_func_preproc.nii.gz', 'rois_ho': '_rois_ho.1D'}


    data_folder = os.path.join(root_folder, 'ABIDE_pcp/cpac/filt_noglobal')
    phenotype = os.path.join(root_folder, 'ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv')
    subject_IDs_path = os.path.join(root_folder, 'subject_IDs.txt')
    Save_ROIs_path = os.path.join(root_folder, 'ABIDE_dir')
    if not os.path.isdir(Save_ROIs_path):
        os.makedirs(Save_ROIs_path)

    abide = datasets.fetch_abide_pcp(data_dir=root_folder, n_subjects=num_subjects, pipeline=pipeline, band_pass_filtering=True, global_signal_regression=False, derivatives=files, legacy_format=False)

    subject_IDs = get_ids(subject_IDs_path)
    labels = get_subject_score(phenotype, subject_IDs, score='DX_GROUP')


    subject_IDs = subject_IDs.tolist()
    filenames = get_filename(subject_IDs, data_folder, filemapping[files[0]])


    data_mat_list = read_timeseries_transfromer_mat(data_folder, Save_ROIs_path, subject_IDs, filenames, 'ho', 'correlation')

    SEX   = get_subject_score(phenotype, subject_IDs, score='SEX')
    AGE   = get_subject_score(phenotype, subject_IDs, score='AGE_AT_SCAN')
    FIQ   = get_subject_score(phenotype, subject_IDs, score='FIQ')
    LABEL = get_subject_score(phenotype, subject_IDs, score='DX_GROUP')
    EYE_S = get_subject_score(phenotype, subject_IDs, score='EYE_STATUS_AT_SCAN')
    BMI   = get_subject_score(phenotype, subject_IDs, score='BMI')

    ADS = []
    Normal = []
    for i in LABEL:
        if LABEL[i] == '1':
            ADS.append(i)
        elif LABEL[i] == '2':
            Normal.append(i)


    print('ADS:', len(ADS))
    print('Normal:', len(Normal))


    sites = get_subject_score(phenotype, subject_IDs, score='SITE_ID')
    unique = np.unique(list(sites.values())).tolist()

    num_classes = 2
    num_nodes = len(subject_IDs)

    y_data = np.zeros([num_nodes, num_classes])
    y = np.zeros([num_nodes, 1])
    site = np.zeros([num_nodes, 1], dtype=int)

    for i in range(num_nodes):
        y_data[i, int(labels[subject_IDs[i]])-1] = 1
        y[i] = int(labels[subject_IDs[i]])
        site[i] = unique.index(sites[subject_IDs[i]])


    features = get_networks(Save_ROIs_path, subject_IDs, kind='correlation', atlas_name='ho')

    label = y
    label = label.reshape(-1)

    skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
    for train_index, test_index in skf.split(features, label):
        pass


    fMRI_feat_256_1 = feature_selection(features, label, train_index, 256)

    fMRI_feat_256_1_pd = pd.DataFrame(fMRI_feat_256_1)
    fMRI_feat_256_1_pd['label'] = label

    fMRI_feat_dict = {}
    for i in range(1):
        fMRI_feat_dict[i] = list(range(i*256,(i+1)*256))


    input_df = pd.read_csv(phenotype, low_memory=False)
    subject_IDs = [int(i) for i in subject_IDs]


    initial_data = input_df[input_df.SUB_ID.isin(subject_IDs)]

    data_age  = initial_data.AGE_AT_SCAN
    data_site = initial_data.SITE_ID
    data_eye = initial_data.EYE_STATUS_AT_SCAN


    data = initial_data.copy()
    min_age, max_age = data.AGE_AT_SCAN.min(), data.AGE_AT_SCAN.max()
    step, bins, block = 2, [min_age], min_age
    while block < max_age:
        block += 2
        bins.append(block)
    data.loc[:,'AGE_AT_SCAN'] = pd.cut(data.AGE_AT_SCAN, bins, right = False)
    data = pd.get_dummies(data, columns=['AGE_AT_SCAN', 'SITE_ID', 'SEX'])


    pheno_list = list(data.columns[data.columns.str.contains('^SITE|^AGE_AT_SCAN|^SEX')])
    anat_list  = ['anat_cnr', 'anat_efc', 'anat_fber', 'anat_fwhm', 'anat_qi1', 'anat_snr']
    func_list  = ['func_efc', 'func_fber', 'func_fwhm', 'func_dvars', 'func_outlier', 'func_quality', 'func_mean_fd', 'func_num_fd', 'func_perc_fd', 'func_gsr']
    feat_list  = pheno_list + anat_list + func_list
    data_func = initial_data[func_list]
    data_anat = initial_data[anat_list]
    data_pheno = data[pheno_list]
    pheno_num = len(pheno_list)
    anat_num  = len(anat_list)
    func_num  = len(func_list)
    ABIDE_three_modal_dict = {'PHENO': pheno_list,
                        'ANAT' : anat_list,
                        'FUNC' : func_list}
    ABIDE_three_modal_num_dict  = {'PHENO': pheno_num,
                            'ANAT' : anat_num,
                            'FUNC' : func_num}
    select_data = data[feat_list]


    standard_list = anat_list + func_list + pheno_list
    scaler        = preprocessing.StandardScaler()
    standard_data = scaler.fit_transform(data[standard_list])
    select_data[standard_list] = standard_data


    np.save(os.path.join(root_folder, 'ABIDE_three_modal_feat_dict.npy'), ABIDE_three_modal_dict)
    select_data.to_csv(os.path.join(root_folder, "ABIDE_three_processed_data_modal.csv"), index = False)

    fMRI_feat_256_1 = pd.read_csv(os.path.join(root_folder, "ABIDE_fMRI_processed_standard_data_256_1.csv"), low_memory=False)
    three_feat_64 = pd.read_csv(os.path.join(root_folder, "ABIDE_three_processed_data_modal.csv"), low_memory=False)


    fMRI_feat_256_1_list = list(fMRI_feat_256_1.columns)[:-1]

    ABIDE_modal_list = ABIDE_three_modal_dict
    ABIDE_modal_list['Correlation'] = fMRI_feat_256_1_list
    DATA_ALL = pd.concat([three_feat_64, fMRI_feat_256_1], axis=1)

    np.save(os.path.join(root_folder, 'ABIDE_modal_feat_dict.npy'),ABIDE_modal_list)
    DATA_ALL.to_csv(os.path.join(root_folder, "ABIDE_processed_data_modal.csv"), index = False)

    print(DATA_ALL.shape)

    if args.popgraph:
        subject_IDs = get_ids(subject_IDs_path)
        graph_feat = create_affinity_graph_from_scores(['SEX', 'SITE_ID'], subject_IDs, phenotype)
        print(f'graph_feat.shape {graph_feat.shape} len(subject_IDs) {len(subject_IDs)}')

        distv = distance.pdist(fMRI_feat_256_1, metric='correlation')
        dist = distance.squareform(distv)
        sigma = np.mean(dist)
        sparse_graph = np.exp(- dist ** 2 / (2 * sigma ** 2))
        final_graph = graph_feat * sparse_graph

        print(f'final_graph.shape {final_graph.shape}')
        np.save(os.path.join(root_folder, 'ABIDE_graph.npy'), final_graph)