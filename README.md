# Multimodal Dataset

This is a repository for multimodal medical data.

# Processing steps

The dataset processing code is collated from [POPGCN](https://github.com/parisots/population-gcn) and [MMGL](https://github.com/SsGood/MMGL).

```cite
@inproceedings{PopGCN,
  title={Spectral graph convolutions for population-based disease prediction},
  author={Parisot, Sarah and Ktena, Sofia Ira and Ferrante, Enzo and Lee, Matthew and Moreno, Ricardo Guerrerro and Glocker, Ben and Rueckert, Daniel},
  booktitle={Medical Image Computing and Computer Assisted Intervention- MICCAI 2017: 20th International Conference},
  pages={177--185},
  year={2017},
  organization={Springer}
}

@article{MMGL,
  title={Multi-modal graph learning for disease prediction},
  author={Zheng, Shuai and Zhu, Zhenfeng and Liu, Zhizhe and Guo, Zhenyu and Liu, Yang and Yang, Yuchen and Zhao, Yao},
  journal={IEEE Transactions on Medical Imaging},
  volume={41},
  number={9},
  pages={2207--2216},
  year={2022},
  publisher={IEEE}
}
```

# Requirements

python==3.8

```shell
numpy        1.24.3
pandas       1.5.2 
nilearn      0.10.2
scikit-learn 1.1.3 
scipy        1.10.1
boto3        1.35.50 
botocore     1.35.50 
tqdm         4.64.1 
```

# ADNI

Alzheimer's Disease Neuroimaging Initiative (ADNI).

## TADPLOE

TADPOLE, a subset of the ADNI database, comprises multimodal data, including magnetic resonance imaging (MRI), positron emission tomography (PET), cognitive tests, cerebrospinal fluid (CSF) biomarkers, risk factors, and demographic information. Subjects with multimodal features were selected and categorized into cognitively normal (CN), Alzheimer's disease (AD), progressive mild cognitive impairment (pMCI), and stable mild cognitive impairment (sMCI) to predict AD progression. 

### Generating Dataset

You need to download  [TADPOLE Challenge data](https://ida.loni.usc.edu/explore/jsp/search/search.jsp?project=ADNI).

```shell
cd TADPOLE

python main.py --select_sub_list ['AD', 'CN', 'SMCI', 'PMCI']  --popgraph
```

## ADNI3

The ADNI3 dataset was screened by ADNI-3, paired MRI(T1-weighted) and PET scan times for each subject (those within three months were grouped together), and combined  the Polygenic Hazard Score (PHS) and demographic information to create a new dataset. The dataset consists of four modalities including MRI, PET and the PHS and risk factors.

The code is still being organized.

## CITE

```cite
@misc{ADNI,
  author = {{Alzheimer’s Disease Neuroimaging Initiative}},
  title = {{Alzheimer’s disease neuroimaging initiative}},
  year = {2019},
  howpublished = {\url{http://ADNI.loni.usc.edu/}}
}

@article{TADPOLE,
  title={TADPOLE challenge: prediction of longitudinal evolution in Alzheimer's disease},
  author={Marinescu, Razvan V and Oxtoby, Neil P and Young, Alexandra L and Bron, Esther E and Toga, Arthur W and Weiner, Michael W and Barkhof, Frederik and Fox, Nick C and Klein, Stefan and Alexander, Daniel C and others},
  journal={ArXiv Preprint ArXiv:1805.03909},
  year={2018}
}

```

# ABIDE

Autism Brain Imaging Data Exchange (ABIDE).

## ABIDE

ABIDE dataset contains over 1,000 functional magnetic resonance imaging (fMRI) scans and corresponding phenotypic data collected from 24 sites. Each subject is characterized by four modalities: demographic information, automated anatomical quality assessment metrics, automated functional quality assessment metrics, and functional MRI connectivity networks. 

```shell
cd ABIDE

python main.py  --popgraph
```

## ABIDE-5

The ABIDE-5 dataset is based on the ABIDE dataset with structural MRI added.

```shell
cd ABIDE-5

python main.py  --popgraph
```

## CITE

```cite
@article{ABIDE,
  title={Deriving reproducible biomarkers from multi-site resting-state data: An Autism-based example},
  author={Abraham, Alexandre and Milham, Michael P and Di Martino, Adriana and Craddock, R Cameron and Samaras, Dimitris and Thirion, Bertrand and Varoquaux, Gael},
  journal={NeuroImage},
  volume={147},
  pages={736--745},
  year={2017},
  publisher={Elsevier}
}
```