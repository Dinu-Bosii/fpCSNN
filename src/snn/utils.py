import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator, GetRDKitFPGenerator
from rdkit.Chem import MACCSkeys
import numpy as np
from csnn_model import CSNNet, train_csnn, val_csnn, test_csnn
from csnn_model_sider import CSNNet as CSNNet2
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, f1_score, precision_score


def load_dataset_df(filename):
    file_path = os.path.join('..', 'data', filename)
    df = pd.read_csv(file_path)
    
    targets = []
    if filename == 'tox21.csv':
        targets = df.columns[0:len(df.columns) - 2]
    
    elif filename == 'sider.csv':
        targets = df.columns[1:]

    return df, targets


def fp_generator(fp_type, fp_size=1024, radius=2):
    fp_type = fp_type.lower()

    if fp_type == 'morgan':
        gen = GetMorganGenerator(radius=radius, fpSize=fp_size)
        def fn(mol, **kwargs):
            return gen.GetFingerprint(mol, **kwargs)

    elif fp_type == 'rdkit':
        gen = GetRDKitFPGenerator(fpSize=fp_size)
        def fn(mol, **kwargs):
            return gen.GetFingerprint(mol, **kwargs)

    elif fp_type == 'maccs':
        def fn(mol, **kwargs):
            return MACCSkeys.GenMACCSKeys(mol, **kwargs)

    return fn


def smile_to_fp(df, fp_config, target_name):
    radius = fp_config['radius']
    mix = fp_config['mix']

    fp1_type, fp1_size = fp_config["fp_type"], fp_config["num_bits"]
    fp1_gen = fp_generator(fp1_type, fp_size=fp1_size, radius=radius)
    array_size = fp1_size

    if mix:
        fp2_type, fp2_size = fp_config["fp_type_2"], fp_config['num_bits_2']
        fp2_gen = fp_generator(fp2_type, fp_size=fp2_size, radius=radius)
        array_size += fp2_size

    num_rows = len(df)

    fp_array = np.zeros((num_rows, array_size))
    target_array = np.zeros((num_rows, 1))

    valid_mols = 0
    for idx, row in df.iterrows():
        mol = Chem.MolFromSmiles(row['smiles'])
        
        if mol is None:
            continue

        fingerprint = np.array(fp1_gen(mol))
        if mix:
            fingerprint_2 = np.array(fp2_gen(mol))
            fingerprint = np.concatenate([fingerprint, fingerprint_2])


        fp_array[valid_mols] = fingerprint
        target_array[valid_mols] = row[target_name]
        valid_mols += 1

    target_array = target_array.ravel()
    fp_array = fp_array[0:valid_mols]
    target_array = target_array[0:valid_mols]

    return fp_array, target_array


def get_spiking_net(net_config):
    input_size = net_config["input_size"]
    time_steps = net_config["time_steps"]
    spike_grad = net_config["spike_grad"]
    beta = net_config["beta"]
    num_outputs = net_config['out_num']
        
    train_fn = train_csnn
    val_fn = val_csnn
    test_fn = test_csnn
    net = None
    if net_config['conv_num'] == 1:
        net = CSNNet(input_size=input_size, num_steps=time_steps, spike_grad=spike_grad, beta=beta, num_outputs=num_outputs)

    elif net_config['conv_num'] == 2:
        net = CSNNet2(input_size=input_size, num_steps=time_steps, spike_grad=spike_grad, beta=beta, num_outputs=num_outputs)
    else:
        raise ValueError(f"Invalid conv_num: {net_config['conv_num']}")

    return net, train_fn, val_fn, test_fn


def calc_metrics(metrics_list, all_targets, all_preds):

    accuracy = accuracy_score(all_targets, all_preds)
    auc_roc = roc_auc_score(all_targets, all_preds)
    tn, fp, fn, tp = confusion_matrix(all_targets, all_preds).ravel()
    sensitivity = tp/(tp + fn)
    specificity = tn/(tn + fp)
    f1 = f1_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds)
    
    print(accuracy, auc_roc)
    metrics_list[0].append(accuracy)
    metrics_list[1].append(auc_roc)
    metrics_list[2].append(sensitivity)
    metrics_list[3].append(specificity)
    metrics_list[4].append(f1)
    metrics_list[5].append(precision)


def make_filename(dirname, target, net_type, fp_config, lr, wd, optim_type, net_config, train_config, net, model = False):
    results_dir = os.path.join("results", dirname, "")
    if model:
        results_dir = os.path.join(results_dir, "models", "")
    spike_grad = net_config['spike_grad']
    
    csnn_channels = f"out-{net.conv1.out_channels}" + (f"-{net.conv2.out_channels}" if hasattr(net, "conv2") else "")
    params = [
        None if dirname == 'BBBP' else target, 
        net_type, 
        f"beta-{net_config['beta']}",
        fp_config['fp_type'],
        None if fp_config['fp_type'] != 'morgan' else 'r-' + f"{fp_config['radius']}",
        fp_config['fp_type_2'] if fp_config['mix'] else None,
        net_config['input_size'],
        csnn_channels,
        f"kernel-{net.conv_kernel}",
        f"stride-{net.conv_stride}",
        f"t{net_config['time_steps']}",
        f"e{train_config['num_epochs']}",
        f"b{train_config['batch_size']}",
        f"lr{lr}",
        train_config['loss_type'],
        optim_type,
        f"wd{wd}",
        None if spike_grad is None else f"sig-{net_config['slope']}",
        "no-bias" if not net_config['bias'] else "bias",
    ]

    filename = results_dir + "_".join(str(p) for p in params if p is not None) + ".csv"
    return filename

