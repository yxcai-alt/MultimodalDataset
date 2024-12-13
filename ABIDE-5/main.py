import os
import argparse
import numpy as np
import boto3
from botocore import UNSIGNED
from botocore.config import Config
import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn import preprocessing
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.feature_selection import RFE
from nilearn import datasets
import csv
import glob
from nilearn import connectome
from concurrent.futures import ThreadPoolExecutor, as_completed
import scipy.io as sio
from scipy.spatial import distance
import warnings
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")


def create_affinity_graph_from_scores(scores, subject_list, phenotype):

    num_nodes = len(subject_list)
    graph = np.zeros((num_nodes, num_nodes))

    for l in scores:
        label_dict = get_subject_score(phenotype, subject_list, l)

        if l in ['AGE_AT_SCAN', 'FIQ']:
            for k in range(num_nodes):
                for j in range(k + 1, num_nodes):
                    try:
                        val = abs(float(label_dict[subject_list[k]]) - float(label_dict[subject_list[j]]))
                        if val < 2:
                            graph[k, j] += 1
                            graph[j, k] += 1
                    except ValueError: 
                        pass

        else:
            for k in range(num_nodes):
                for j in range(k + 1, num_nodes):
                    if label_dict[subject_list[k]] == label_dict[subject_list[j]]:
                        graph[k, j] += 1
                        graph[j, k] += 1

    return graph



def feature_selectionv2(matrix, labels, train_ind, fnum):

    estimator = RidgeClassifier()
    selector = RFE(estimator, n_features_to_select=fnum, step=100, verbose=1)

    featureX = matrix.iloc[train_ind, :]
    featureY = labels[train_ind]
    selector = selector.fit(featureX, featureY.ravel())
    x_data = selector.transform(matrix)
    x_data = pd.DataFrame(selector.transform(matrix), columns=matrix.columns[selector.support_])

    X_train, X_test, y_train, y_test = train_test_split(x_data, labels, test_size=0.2, random_state=42)

    estimator.fit(X_train, y_train.ravel())

    predictions = estimator.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("Number of labeled samples %d" % len(train_ind))
    print("Number of features selected %d" % x_data.shape[1])
    print("Model accuracy: %.2f%%" % (accuracy * 100))
    
    return x_data


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


def read_timeseries_transfromer_mat(folder_path, subject_IDs, filenames_list, atlas_name, kind):

    data_mat_list = []
    for file_name_, subject_ID_ in zip(filenames_list, subject_IDs):

        ROIs_file_path = os.path.join(folder_path, file_name_)

        timeseries_ = np.loadtxt(ROIs_file_path, skiprows=0)
        if kind in ['tangent', 'partial correlation', 'correlation']:
            conn_measure = connectome.ConnectivityMeasure(kind=kind)
            connectivity = conn_measure.fit_transform([timeseries_])[0]  
        data_mat_list.append(connectivity)
    return data_mat_list


def get_networks(data_mat_list):


    idx = np.triu_indices_from(data_mat_list[0], 1)  
    norm_networks = [np.arctanh(mat) for mat in data_mat_list]
    vec_networks = [mat[idx] for mat in norm_networks]
    matrix = np.vstack(vec_networks)

    return matrix


def get_url_list(filenames, str_):
    
    url_list = []
    for filename in filenames:
        index = filename.replace(str_, "")
        url_list.append(f'data/Projects/ABIDE_Initiative/Outputs/freesurfer/5.1/{index}/stats/')

    return url_list


def download_file(bucket_name, prefix, file_name, save_folder):
    s3_key = os.path.join(prefix, file_name)
    local_file_path = os.path.join(save_folder, file_name)
    if os.path.exists(local_file_path):
        print(f"{file_name} already exists, skip download")
        return
    try:
        s3.download_file(bucket_name, s3_key, local_file_path)
        print(f"Downloaded {file_name} to {local_file_path}")
    except Exception as e:
        print(f"Failed to download {file_name}: {e}")


def download_files_from_s3(bucket_name, prefix, required_files, save_folder):
    with ThreadPoolExecutor(max_workers=10) as executor:  
        futures = [executor.submit(download_file, bucket_name, prefix, file_name, save_folder) for file_name in required_files]
        for future in as_completed(futures):
            future.result()  


def DF_to_1(df):

    df = df.set_index('StructName')


    df = df.stack()
    df.index = [f"{idx[0]}_{idx[1]}" for idx in df.index]


    df = df.to_frame().T

    return df


def read_wmparc_stats(file_path):

    data = []
    with open(file_path, 'r') as file:
        for line in file:

            if line.startswith('#'):
                continue
            
            parts = line.split()
            if len(parts) >= 10:  
                index = int(parts[0])
                seg_id = int(parts[1])
                nvoxels = int(parts[2])
                volume = float(parts[3])
                struct_name = parts[4]
                norm_mean = float(parts[5])
                norm_stddev = float(parts[6])
                norm_min = float(parts[7])
                norm_max = float(parts[8])
                norm_range = float(parts[9])
                

                data.append([index, seg_id, nvoxels, volume, struct_name, norm_mean, norm_stddev, norm_min, norm_max, norm_range])


    df = pd.DataFrame(data, columns=['Index', 'SegId', 'NVoxels', 'Volume_mm3', 'StructName', 'normMean', 'normStdDev', 'normMin', 'normMax', 'normRange'])

    return df[['StructName', 'Volume_mm3', 'normMean']]


def read_entorhinal_exvivo_stats(file_path):

    data = []
    with open(file_path, 'r') as file:
        for line in file:

            if line.startswith('#'):
                continue


            parts = line.split()
            if len(parts) >= 10:  
                struct_name = parts[0]
                num_vert = int(parts[1])
                surf_area = int(parts[2])    
                gray_vol = int(parts[3])     
                thick_avg = float(parts[4])  
                thick_std = float(parts[5])  


                data.append([struct_name, surf_area, gray_vol, thick_avg, thick_std])


    df = pd.DataFrame(data, columns=['StructName', 'SurfArea', 'GrayVol', 'ThickAvg', 'ThickStd'])


    return df[['StructName', 'SurfArea', 'GrayVol', 'ThickAvg']]


def Read_aparc_stats(file_path):

    data = []
    with open(file_path, 'r') as file:
        for line in file:

            if line.startswith('#'):
                continue
            
            parts = line.split()
            if len(parts) >= 10:  
                struct_name = parts[0]
                surf_area = int(parts[2])    
                gray_vol = int(parts[3])     
                thick_avg = float(parts[4])  


                data.append([struct_name, surf_area, gray_vol, thick_avg])


    df = pd.DataFrame(data, columns=['StructName', 'SurfArea', 'GrayVol', 'ThickAvg'])


    return df[['StructName', 'SurfArea', 'GrayVol', 'ThickAvg']]


def Read_aseg_stats(file_path):

    data = []
    with open(file_path, 'r') as file:
        for line in file:

            if line.startswith('#'):
                continue
            
            parts = line.split()
            if len(parts) >= 10:  
                index = int(parts[0])
                seg_id = int(parts[1])
                nvoxels = int(parts[2])
                volume = float(parts[3])         
                struct_name = parts[4]
                norm_mean = float(parts[5])      
                norm_stddev = float(parts[6])    
                

                data.append([struct_name, nvoxels, volume, norm_mean])


    df = pd.DataFrame(data, columns=['StructName', 'NVoxels', 'Volume_mm3', 'ThickAvg'])


    return df[['StructName', 'Volume_mm3', 'ThickAvg']]



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

    abide = datasets.fetch_abide_pcp(data_dir=root_folder, n_subjects=num_subjects, pipeline=pipeline, band_pass_filtering=True, global_signal_regression=False, derivatives=files, legacy_format=False)

    subject_IDs_path = os.path.join(root_folder, 'subject_IDs.txt')
    subject_IDs = get_ids(subject_IDs_path)
    phenotype = os.path.join(root_folder, 'ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv')
    labels = get_subject_score(phenotype, subject_IDs, score='DX_GROUP')

    data_folder = os.path.join(root_folder, 'ABIDE_pcp/cpac/filt_noglobal')
    filenames = get_filename(subject_IDs, data_folder, filemapping[files[0]])

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


    data_mat_list = read_timeseries_transfromer_mat(data_folder, subject_IDs, filenames, 'ho', 'correlation')
    matrix = get_networks(data_mat_list)




    label = y
    label = label.reshape(-1)
    features = matrix

    skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
    for train_index, test_index in skf.split(features, label):
        print(f'{len(train_index)}, {len(test_index)}')



    fMRI_feat_256_1 = feature_selection(features, label, train_index, 256)
    fMRI_feat_256_1_pd = pd.DataFrame(fMRI_feat_256_1)

    scaler = StandardScaler()
    fMRI_feat_256_1 = scaler.fit_transform(fMRI_feat_256_1)

    fMRI_feat_256_1_pd['label'] = label
    fMRI_feat_256_1_pd.to_csv(os.path.join(root_folder, "ABIDE_fMRI_processed_standard_data_256_1.csv"), index = False)


    columns = [f'fmri_{i+1}' for i in range(fMRI_feat_256_1.shape[1])]
    fMRI_feat_256_1_df = pd.DataFrame(fMRI_feat_256_1, columns=columns)


    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

    bucket_name = 'fcp-indi'
    prefixes = get_url_list(filenames, filemapping[files[0]])


    required_files = ['aseg.stats',
    'lh.BA.stats',
    'lh.aparc.a2009s.stats',
    'lh.aparc.stats',
    'lh.entorhinal_exvivo.stats',
    'rh.BA.stats',
    'rh.aparc.a2009s.stats',
    'rh.aparc.stats',
    'rh.entorhinal_exvivo.stats',
    'wmparc.stats']



    for ids, prefix in zip(subject_IDs.tolist(), prefixes):
        SAVE_FOLDER_MRI = os.path.join(root_folder, 'ADIDE_MRI_data', ids)
        if not os.path.exists(SAVE_FOLDER_MRI):
            os.makedirs(SAVE_FOLDER_MRI)
        download_files_from_s3(bucket_name, prefix, required_files, SAVE_FOLDER_MRI)


    print('='*20 + "MRI PRO" + '='*20)
    for ids, prefix in tqdm(zip(subject_IDs, prefixes), total=len(subject_IDs)):
        file_path_ = os.path.join(root_folder, 'ADIDE_MRI_data', ids)
        if not os.path.exists(os.path.join(file_path_, 'MRI.csv')):
            wmparc_file_path = os.path.join(file_path_, 'wmparc.stats')
            DF_wmparc = read_wmparc_stats(wmparc_file_path)
            DF_wmparc_1 = DF_to_1(DF_wmparc)


            RH_file_path = os.path.join(file_path_, 'rh.entorhinal_exvivo.stats')
            DF_RH_entorhinal_exvivo = read_entorhinal_exvivo_stats(RH_file_path)
            DF_RH_entorhinal_exvivo_1 = DF_to_1(DF_RH_entorhinal_exvivo)
            DF_RH_entorhinal_exvivo_1 = DF_RH_entorhinal_exvivo_1.add_prefix('RH')


            LH_file_path = os.path.join(file_path_, 'Lh.entorhinal_exvivo.stats')
            DF_LH_entorhinal_exvivo = read_entorhinal_exvivo_stats(LH_file_path)
            DF_LH_entorhinal_exvivo_1 = DF_to_1(DF_LH_entorhinal_exvivo)
            DF_LH_entorhinal_exvivo_1 = DF_LH_entorhinal_exvivo_1.add_prefix('LH')


            RH_file_path = os.path.join(file_path_, 'rh.aparc.a2009s.stats')
            DF_RH_aparc_stats = Read_aparc_stats(RH_file_path)
            DF_RH_aparc_stats_1 = DF_to_1(DF_RH_aparc_stats)
            DF_RH_aparc_stats_1 = DF_RH_aparc_stats_1.add_prefix('RH')


            LH_file_path = os.path.join(file_path_, 'Lh.aparc.a2009s.stats')
            DF_LH_aparc_stats = Read_aparc_stats(LH_file_path)
            DF_LH_aparc_stats_1 = DF_to_1(DF_LH_aparc_stats)
            DF_LH_aparc_stats_1 = DF_LH_aparc_stats_1.add_prefix('LH')


            aseg_file_path = os.path.join(file_path_, 'aseg.stats')
            DF_aseg_stats = Read_aseg_stats(aseg_file_path)
            DF_aseg_stats_1 = DF_to_1(DF_aseg_stats)

            DF_MRI = pd.concat([DF_aseg_stats_1, DF_LH_entorhinal_exvivo_1, DF_RH_entorhinal_exvivo_1, DF_LH_aparc_stats_1, DF_RH_aparc_stats_1, DF_wmparc_1], axis=1)
            DF_MRI.to_csv(os.path.join(file_path_, 'MRI.csv'), index=False)

    print('='*20 + "MRI Loading" + '='*20)
    DF_MRI_List = []
    for ids, prefix in tqdm(zip(subject_IDs, prefixes), total=len(subject_IDs)):
        file_path_ = os.path.join(root_folder, 'ADIDE_MRI_data', ids, 'MRI.csv')
        df = pd.read_csv(file_path_)
        DF_MRI_List.append(df)


    MRI_DF = pd.concat(DF_MRI_List, axis=0, ignore_index=True)
    MRI_DF = MRI_DF.dropna()
    CLEANED_INDEX = MRI_DF.index.tolist()

    label = y
    label = np.squeeze(label, -1)[CLEANED_INDEX]


    scaler = StandardScaler()
    normalized_matrix = scaler.fit_transform(MRI_DF)
    normalized_matrix = pd.DataFrame(normalized_matrix, columns=MRI_DF.columns)

    skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
    for train_index, test_index in skf.split(MRI_DF, label):
        print(f'{len(train_index)}, {len(test_index)}')



    MRI_feat_150_1_pd = feature_selectionv2(normalized_matrix, label, train_index, 150)
    MRI_feat_150_1_pd.to_csv(os.path.join(root_folder, "ABIDE_MRI_processed_standard_data_150_1.csv"), index = False)


    AGE_AT_SCAN_ = get_subject_score(phenotype, subject_IDs, score='AGE_AT_SCAN')
    SITE_ID_ = get_subject_score(phenotype, subject_IDs, score='SITE_ID')
    SEX_ = get_subject_score(phenotype, subject_IDs, score='SEX')


    Origin_dict = {'AGE_AT_SCAN':[], 'SITE_ID':[], 'SEX':[]}
    for i in subject_IDs:
        Origin_dict['AGE_AT_SCAN'].append(float(AGE_AT_SCAN_[i]))
        Origin_dict['SITE_ID'].append(SITE_ID_[i])
        Origin_dict['SEX'].append(float(SEX_[i]))


    Origin_dict_pd = pd.DataFrame(Origin_dict)
    Origin_dict_pd['SITE_ID'] = pd.factorize(Origin_dict_pd['SITE_ID'])[0]
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

    ABIDE_modal_dict = ABIDE_three_modal_dict
    ABIDE_modal_dict['MRI'] = MRI_feat_150_1_pd.columns.tolist()
    ABIDE_modal_dict['fMRI'] = fMRI_feat_256_1_df.columns.tolist()


    select_data = select_data.iloc[CLEANED_INDEX,:]
    fMRI_feat_256_1_df = fMRI_feat_256_1_df.iloc[CLEANED_INDEX,:]

    Origin_dict_pd = Origin_dict_pd.iloc[CLEANED_INDEX,:]
    # Origin_dict_pd.to_csv(os.path.join(root_folder, "ABIDE_Origin.csv"), index = False)

    label = y
    label = np.squeeze(label, -1)[CLEANED_INDEX]

    select_data.reset_index(drop=True, inplace=True)
    MRI_feat_150_1_pd.reset_index(drop=True, inplace=True)
    fMRI_feat_256_1_df.reset_index(drop=True, inplace=True)


    DATA_ALL = pd.concat([select_data, MRI_feat_150_1_pd, fMRI_feat_256_1_df], axis=1)
    DATA_ALL['label'] = label

    np.save(os.path.join(root_folder, 'ABIDE_modal_feat_dict.npy'), ABIDE_modal_dict)
    DATA_ALL.to_csv(os.path.join(root_folder, "ABIDE_processed_data_modal.csv"), index = False)

    print(DATA_ALL.shape)

    ADS = []
    Normal = []
    for i in label:
        if i == 1:
            ADS.append(i)
        elif i == 2:
            Normal.append(i)

    print('ADS:', len(ADS))
    print('Normal:', len(Normal))

    if args.popgraph:
        subject_IDs = get_ids(subject_IDs_path)
        subject_IDs = subject_IDs[CLEANED_INDEX]
        graph_feat = create_affinity_graph_from_scores(['SEX', 'SITE_ID'], subject_IDs, phenotype)
        print(f'graph_feat.shape {graph_feat.shape} len(subject_IDs) {len(subject_IDs)}')


        distv = distance.pdist(fMRI_feat_256_1_df, metric='correlation')
        dist = distance.squareform(distv)
        sigma = np.mean(dist)
        sparse_graph = np.exp(- dist ** 2 / (2 * sigma ** 2))
        final_graph = graph_feat * sparse_graph

        print(f'final_graph.shape {final_graph.shape}')
        np.save(os.path.join(root_folder, 'ABIDE_graph.npy'), final_graph)
        with open(os.path.join(root_folder, 'subject_ids_mri.txt'), 'w') as file:
            for id in subject_IDs.tolist():
                file.write(id + '\n')
