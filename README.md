## fpCSNN: Convolutional Spiking Neural Networks with Molecular Fingerprints for Molecular Property Prediction 

In this work, we propose a Convolutional Spiking Neural Network (SNN), fpCSNN, for predicting molecular properties using molecular fingerprints. The model is trained with backpropagation and utilizes Leaky Integrate-and-Fire (LIF) neurons, with rate coding to process the output spike trains.

![ScreenShot](figures/csnn-mol.png?raw=true)
## Data Availability and Preprocessing

The datasets used in this work (Tox21 and SIDER) are publicly available at the [MoleculeNet Repository](https://moleculenet.org/datasets-1). The SMILES strings are converted to Molecular Fingerprints using [RDKit](https://www.rdkit.org/).

## Requirements
This work relies on the PyTorch and snnTorch frameworks to implement, train, and test the proposed model. It was tested with Python 3.10.0 and CUDA 12.4. The following packages are required:

```
rdkit==2024.9.4
scikit-learn==1.6.1
torch==2.6.0+cu124
snntorch==0.9.1
xgboost==2.1.4
numpy==2.2.2
pandas==2.2.3

```
## References
[1] Eshraghian, J. K., Ward, M., Neftci, E., Wang, X., Lenz, G., Dwivedi, G., Bennamoun, M., Jeong, D. S., & Lu, W. D.: Training spiking neural networks using lessons from deep learning. *Proceedings of the IEEE, 111*(9), 1016–1054, (2023).
```
@article{eshraghian2021training,
        title   =  {Training spiking neural networks using lessons from deep learning},
        author  =  {Eshraghian, Jason K and Ward, Max and Neftci, Emre and Wang, Xinxin
                        and Lenz, Gregor and Dwivedi, Girish and Bennamoun, Mohammed and
                        Jeong, Doo Seok and Lu, Wei D},
        journal = {Proceedings of the IEEE},
        volume  = {111},
        number  = {9},
        pages   = {1016--1054},
        year    = {2023}
}
```

[2] Xie, L., Xu, L., Kong, R., Chang, S., & Xu, X.: Improvement of Prediction Performance With Conjoint Molecular Fingerprint in Deep Learning. *Frontiers in Pharmacology, 11*, (2020). [https://doi.org/10.3389/fphar.2020.606668](https://doi.org/10.3389/fphar.2020.606668)
```
@article{10.3389/fphar.2020.606668,
        author   = {Xie, Liangxu  and Xu, Lei  and Kong, Ren  and Chang, Shan  and Xu, Xiaojun },
        title    = Improvement of Prediction Performance With Conjoint Molecular Fingerprint in Deep Learning},
        jounrnal = {Frontiers in Pharmacology},
        volume   = {11},
        year     = { 2020},
        url      = { https://www.frontiersin.org/journals/pharmacology/articles/10.3389/fphar.2020.606668},
        doi      = {10.3389/fphar.2020.606668},
        issn     = { 1663-9812},
        abstract = {The accurate predicting of physical properties and bioactivity of drug molecules in deep learning depends on how molecules are represented. Many types of molecular descriptors have been developed for quantitative structure-activity/property relationships quantitative structure-activity relationships (QSPR). However, each molecular descriptor is optimized for a specific application with encoding preference. Considering that standalone featurization methods may only cover parts of information of the chemical molecules, we proposed to build the conjoint fingerprint by combining two supplementary fingerprints. The impact of conjoint fingerprint and each standalone fingerprint on predicting performance was systematically evaluated in predicting the logarithm of the partition coefficient (logP) and binding affinity of protein-ligand by using machine learning/deep learning (ML/DL) methods, including random forest (RF), support vector regression (SVR), extreme gradient boosting (XGBoost), long short-term memory network (LSTM), and deep neural network (DNN). The results demonstrated that the conjoint fingerprint yielded improved predictive performance, even outperforming the consensus model using two standalone fingerprints among four out of five examined methods. Given that the conjoint fingerprint scheme shows easy extensibility and high applicability, we expect that the proposed conjoint scheme would create new opportunities for continuously improving predictive performance of deep learning by harnessing the complementarity of various types of fingerprints.}
}
```
