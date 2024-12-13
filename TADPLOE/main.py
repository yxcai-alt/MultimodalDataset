import os
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn import preprocessing
from sklearn.feature_selection import RFE
from sklearn.svm import LinearSVC
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from itertools import chain
from scipy.spatial import distance
import argparse


def modal_extraction(feat_dict):

    feat_dict = feat_dict[~feat_dict['FLDNAME'].str.contains('RID|SID|UID|PTID|VISCODE|DATE|FLDSTRENG|FLDSTRENG|VERSION|STATUS|NAME|VISIT')]
    feat_modal = feat_dict['TBLNAME']
    modal = feat_modal.drop_duplicates(keep = 'last')
    modal = modal[modal != " "]
    modal = list(modal)
    detached_feat = []
    for i in modal:
        temp = list(feat_dict[feat_modal == i]['FLDNAME'])  
        detached_feat.append(temp)
    modal_feat = dict(zip(modal, detached_feat))  
    
    return modal_feat


def discrete_feature_processing(data, m_d, remove_longitudinal_mri = True, PMCI_selection = False):
    delete = set(['OVERALLQC_UCSFFSX_11_02_15_UCSFFSX51_08_01_16', 
                                           'TEMPQC_UCSFFSX_11_02_15_UCSFFSX51_08_01_16', 
                                           'FRONTQC_UCSFFSX_11_02_15_UCSFFSX51_08_01_16',
                                           'PARQC_UCSFFSX_11_02_15_UCSFFSX51_08_01_16',
                                           'INSULAQC_UCSFFSX_11_02_15_UCSFFSX51_08_01_16',
                                           'OCCQC_UCSFFSX_11_02_15_UCSFFSX51_08_01_16',
                                           'BGQC_UCSFFSX_11_02_15_UCSFFSX51_08_01_16',
                                           'CWMQC_UCSFFSX_11_02_15_UCSFFSX51_08_01_16',
                                           'VENTQC_UCSFFSX_11_02_15_UCSFFSX51_08_01_16'])
    m_d['UCSFFSX'] = list(set(m_d['UCSFFSX']) - delete)
    m_d['UPENNBIOMK9'] = ['ABETA_UPENNBIOMK9_04_19_17', 'TAU_UPENNBIOMK9_04_19_17', 'PTAU_UPENNBIOMK9_04_19_17']
    data.loc[data['ABETA_UPENNBIOMK9_04_19_17'] == '<200', 'ABETA_UPENNBIOMK9_04_19_17'] = 200
    data.loc[data['TAU_UPENNBIOMK9_04_19_17'] == '<80', 'TAU_UPENNBIOMK9_04_19_17'] = 80
    data.loc[data['TAU_UPENNBIOMK9_04_19_17'] == '>1300', 'TAU_UPENNBIOMK9_04_19_17'] = 1300
    data.loc[data['PTAU_UPENNBIOMK9_04_19_17'] == '<8', 'PTAU_UPENNBIOMK9_04_19_17'] = 8
    data.loc[data['PTAU_UPENNBIOMK9_04_19_17'] == '>120', 'PTAU_UPENNBIOMK9_04_19_17'] = 120

    min_age, max_age = data.AGE.min(), data.AGE.max()
    step, bins, block = 2, [min_age], min_age
    while block < max_age:
        block += 2
        bins.append(block)
    data.loc[:,'AGE'] = pd.cut(data.AGE, bins, right = False)
    data = pd.get_dummies(data, columns=['AGE'])
    
    data.loc[data.PTGENDER == 'Male', 'PTGENDER'] = 1
    data.loc[data.PTGENDER == 'Female', 'PTGENDER'] = 0
    
    data = pd.get_dummies(data, columns = ['PTMARRY'])

    min_educat, max_educat = data.PTEDUCAT.min(), data.PTEDUCAT.max()
    step, bins, block = 2, [min_educat], min_educat
    while block < max_educat:
        block += 2
        bins.append(block)
    data.loc[:,'PTEDUCAT'] = pd.cut(data.PTEDUCAT, bins, right = False)
    data = pd.get_dummies(data, columns=['PTEDUCAT'])
    
    data = pd.get_dummies(data, columns=['APOE4'])
    
    RISK_FACTOR = list(data.columns[data.columns.str.contains('AGE_|PTMARRY|PTEDUCAT|APOE4|PTGENDER')])
    if PMCI_selection == False:
        COGNITIVE_TEST = list(data.columns[data.columns.str.contains('CDRSB|ADAS|MMSE|RAVLT|FAQ|MOCA|Ecog')])
        ROI_AVERAGE = list(data.columns[data.columns.str.contains('^FDG$|^AV45$|^Ventricles$|^Hippocampus$|^WholeBrain$|^Entorhinal$|^Fusiform$|^MidTemp$|^ICV$')])
        
    else:
        COGNITIVE_TEST = list(data.columns[data.columns.str.contains('CDRSB|ADAS|MMSE|RAVLT|FAQ')])
        ROI_AVERAGE = list(data.columns[data.columns.str.contains('^Ventricles$|^Hippocampus$|^WholeBrain$|^Entorhinal$|^Fusiform$|^MidTemp$|^ICV$')])
    
    m_d['RISK_FACTOR'] = RISK_FACTOR
    m_d['COGNITIVE_TEST'] = COGNITIVE_TEST
    m_d['ROI_AVERAGE'] = ROI_AVERAGE
    m_d.pop('ADNIMERGE')
    if remove_longitudinal_mri:
        m_d.pop('UCSFFSL')
    data.rename(columns={"DXCHANGE": "LABEL"}, inplace = True)
    
    return data, m_d


def sample_selection(data, based_feat):

    data_x = data[based_feat]
    temp = ~data_x.isnull()
    index = temp.all(axis='columns')
    data_non_null = data[index]
    
    return data_non_null


def feature_selection(data_x, feat_dict, modal, num_features_selected):

    set_modal_feat = set(feat_dict[modal])
    set_data_feat = set(data_x.columns)
    feat_list = list(set_data_feat.intersection(set_modal_feat))
    feat_list.append('LABEL')
    
    print('The number of features of intersection between modal and data:', len(feat_list))
    
    data_modal = data_x[feat_list]
    data = sample_selection(data_modal, feat_list)
    
    data_X = data[data.columns.drop('LABEL')]
    
    MAP_dict_L = dict(zip(data.LABEL.unique().tolist(), range(len(data.LABEL.unique().tolist()))))
    data_Y = data['LABEL'].map(MAP_dict_L)
    
    
    if len(data_X) != 0:

        print('The number of samples applicable to the feature selection condition:', len(data_X))
        scaler = preprocessing.StandardScaler()
        input_data_X = scaler.fit_transform(data_X)
        estimator = RidgeClassifier()
        selector = RFE(estimator, n_features_to_select=num_features_selected, step=200, verbose=0)
        selector = selector.fit(input_data_X, data_Y)
        selected_feat = list(np.array(data_X.columns)[selector.support_])
        return selected_feat
    else:
        print('There is no sample has all the features of Modal '+ modal)
        return []
    

def parse_args():
    parser = argparse.ArgumentParser(description='TADPOLE data processing script')
    parser.add_argument('--select_sub_list', nargs='+', default=['AD', 'CN', 'SMCI', 'PMCI'], help='List of selected subjects')
    parser.add_argument('--popgraph', action='store_true', help='Enable feature selection')
    return parser.parse_args()


def create_affinity_graph_from_gender_age(gender_data, age_data):
    num_nodes = len(gender_data)
    graph = np.zeros((num_nodes, num_nodes))

    for k in range(num_nodes):
        for j in range(k + 1, num_nodes):
            if gender_data[k] == gender_data[j]:
                graph[k, j] += 1
                graph[j, k] += 1

    for k in range(num_nodes):
        for j in range(k + 1, num_nodes):
            val = abs(age_data[k] - age_data[j])
            if val < 2:
                graph[k, j] += 1
                graph[j, k] += 1

    return graph


if __name__ == '__main__':
    args = parse_args()
    Select_Sub_list = args.select_sub_list

    root_folder = os.getcwd()
    # CN AD SMCI PMCI 
    # Select_Sub_list = ['SMCI', 'PMCI']
    # Select_Sub_list = ['AD', 'CN', 'SMCI', 'PMCI']
    # Select_Sub_list = ['AD', 'CN', 'SMCI']


    DATA_LABEL_List = ['AD', 'CN', 'SMCI', 'PMCI']
    if 'PMCI' in Select_Sub_list:
        PMCI_selection = True
    else:
        PMCI_selection = False
        DATA_LABEL_List.remove('PMCI')

    LABEL_ENCODER = {'AD':3, 'CN':1, 'SMCI':2, 'PMCI':5}
    Save_NO_process_data_feature_list = ['PTGENDER', 'SITE', 'AGE', 'MMSE', 'APOE4', 'PTEDUCAT', 'PTMARRY']

    input_dict_df = pd.read_csv(os.path.join(root_folder, 'tadpole_challenge/TADPOLE_D1_D2_Dict.csv'))
    input_df = pd.read_csv(os.path.join(root_folder, 'tadpole_challenge/TADPOLE_D1_D2.csv'),low_memory=False)
    input_dict_df.loc[np.where(input_dict_df.FLDNAME.values=='DXCHANGE')[0][0]].TEXT

    feature_list = input_df.columns
    feature_list = feature_list[~feature_list.str.contains('_bl$')]
    input_df = input_df[feature_list]

    feature_list = feature_list[~feature_list.str.upper().str.contains('UCSFFSL')]
    input_df = input_df[feature_list]

    cond = input_df != " "
    input_df = input_df.where(cond, np.nan)

    feat_nan = []
    for feat in feature_list:
        if input_df[feat].isnull().all() == True:
            feat_nan.append(feat)
    input_df.drop(feat_nan,axis=1, inplace=True)

    Select_list = []
    for Label_Name in DATA_LABEL_List:
        Select_list.append(input_df[input_df.DXCHANGE == LABEL_ENCODER[Label_Name]])

    data = pd.concat(Select_list)

    index = data.VISCODE.str.contains('^bl$|^m06$|^m12$|^m18$|^m24$|^m30$|^m36$|^m42$|^m48$')
    data = data[index]
    data_baseline = data[data.VISCODE == 'bl']

    assert data_baseline.shape == data_baseline.drop_duplicates(subset=['RID']).shape

    feat_dict = modal_extraction(input_dict_df)

    data_processed, feature_dict_processed = discrete_feature_processing(data, feat_dict, remove_longitudinal_mri = True, PMCI_selection = PMCI_selection)


    selected_feat = {}
    for modal_name in feature_dict_processed.keys():
        print('feature selection for modal: '+ modal_name)
        selected_feat_ = feature_selection(data_processed, feature_dict_processed, modal_name, 150)
        selected_feat[modal_name] = selected_feat_
        print('--------------------------------------------------------------')


    feat_sam_sel = selected_feat['COGNITIVE_TEST'] + selected_feat['ROI_AVERAGE'] + selected_feat['RISK_FACTOR'] + selected_feat['UPENNBIOMK9']
    data = sample_selection(data_processed, feat_sam_sel)
    sample_num, feat_num = data.shape
    nan_counts = data.apply(lambda x: x.isnull().value_counts())
    nan_counts[nan_counts.isnull()] = 0
    valid_feature = nan_counts.iloc[[1]] < int(sample_num * 0.15)
    avail_feat_set = set(nan_counts.columns[valid_feature.values[0]])
    for i in selected_feat:
        if selected_feat[i] != None:
            selected_feat[i] = list(set(selected_feat[i]).intersection(avail_feat_set))

    selected_feat.pop('BAIPETNMRC')
    selected_feat.pop('UCBERKELEYAV1451')
    selected_feat.pop('DTIROI')
    if PMCI_selection == True:
        selected_feat.pop('UCBERKELEYAV45')

    feat_all = []
    for i in list(selected_feat.values()):
        feat_all += i
    feat_all += ['LABEL']
    feat_all += ['RID']

    final_data = data[feat_all]
    data = final_data
    data_rm_repeat = data.drop_duplicates(subset=['RID'], keep='last')


    Final_LABEL_list = []
    for LABEL_ in Select_Sub_list:
        Final_LABEL_list.append(LABEL_ENCODER[LABEL_])


    data_rm_repeat = data_rm_repeat[data_rm_repeat['LABEL'].isin(Final_LABEL_list)]
    MAP_dict = dict(zip(data_rm_repeat.LABEL.unique().tolist(), range(1, len(data_rm_repeat.LABEL.unique().tolist()) + 1)))
    data_rm_repeat['LABEL'] = data_rm_repeat['LABEL'].map(MAP_dict)
    data_rm_repeat['LABEL'].value_counts()

    RID_list = data_rm_repeat.RID.values

    feat_all.remove('RID')
    feat_all.remove('LABEL')

    feat_list = feat_all

    scaler = preprocessing.StandardScaler()
    standard_data = scaler.fit_transform(data_rm_repeat[feat_list])
    data_rm_repeat[feat_list] = standard_data

    data = data_rm_repeat.drop('RID',axis=1)


    data = data.astype(dtype='float')
    nan_index = data.isnull().sum()
    index = nan_index.index
    for i in index:
        if nan_index[i] != 0:
            data[i].fillna(data[i].mean(), inplace=True)

    print('modal number: {} featuer number: {}'.format(len(selected_feat.keys()),len(list(chain.from_iterable(selected_feat.values())))))


    LABEL_ENCODER_reversed = {v: k for k, v in LABEL_ENCODER.items()}
    for val_name, val_ in MAP_dict.items():
        print('LABEL: {}  Number: {}'.format(LABEL_ENCODER_reversed[val_name], data.LABEL.value_counts()[val_]))


    if PMCI_selection == False:
        np.save(os.path.join(root_folder, 'ADNI_modal_feat_dict.npy'), selected_feat)
        data.to_csv(os.path.join(root_folder, "ADNI_processed_standard_data.csv"), index = False)
    else:

        np.save(os.path.join(root_folder, 'ADNI_modal_feat_dict.npy'),selected_feat)
        data.to_csv(os.path.join(root_folder, "ADNI_processed_standard_data.csv"), index = False)
        
    print('modalities: ', selected_feat.keys())
    print('preprocessed data shape: ', data.shape)

    if args.popgraph:
        gender_data = data_processed.PTGENDER[data.index].values
        age_data = input_df.AGE[data.index].values
        graph_feat = create_affinity_graph_from_gender_age(gender_data, age_data)
        MRI_data = data.loc[:,selected_feat['UCSFFSX']]
        distv = distance.pdist(MRI_data, metric='correlation')
        dist = distance.squareform(distv)
        sigma = np.mean(dist)
        sparse_graph = np.exp(- dist ** 2 / (2 * sigma ** 2))
        final_graph = graph_feat * sparse_graph
        print(f'final_graph.shape {final_graph.shape}')
        np.save(os.path.join(root_folder, 'ADNI_graph.npy'), final_graph)