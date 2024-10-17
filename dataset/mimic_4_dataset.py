import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from sklearn.preprocessing import StandardScaler
from models.mimic.trans_v2 import Transformer, Transformer_rejector, Custom
import torch.nn as nn
import numpy as np
tmp_store = 0


def categorize_los(days: int):
    """Categorizes length of stay into 10 categories.

    One for ICU stays shorter than a day, seven day-long categories for each day of
    the first week, one for stays of over one week but less than two,
    and one for stays of over two weeks.

    Args:
        days: int, length of stay in days

    Returns:
        category: int, category of length of stay
    """
    # ICU stays shorter than a day
    if days < 1:
        return 0
    # else:
    #     return days
    # each day of the first week
    elif 1 <= days <= 7:
        return days
    # stays of over one week but less than two
    elif 7 < days <= 14:
        return 8
    # stays of over two weeks
    else:
        return 9

def mortality_prediction_lenght_stay(patient):
    """
    patient is a <pyhealth.data.Patient> object
    """
    global tmp_store
    samples = []
    # loop over all visits but the last one
    for i in range(len(patient) - 1):

        # visit and next_visit are both <pyhealth.data.Visit> objects
        visit = patient[i]
        next_visit = patient[i + 1]

        # step 1: define the mortality_label
        if next_visit.discharge_status not in [0, 1]:
            mortality_label = 0
        else:
            mortality_label = int(next_visit.discharge_status)

        # step 2: get code-based feature information
        conditions = visit.get_code_list(table="DIAGNOSES_ICD")
        procedures = visit.get_code_list(table="PROCEDURES_ICD")
        drugs = visit.get_code_list(table="PRESCRIPTIONS")

        # step 3: exclusion criteria: visits without condition, procedure, or drug
        if len(conditions) * len(procedures) * len(drugs) == 0: continue
        if (mortality_label == 0 and tmp_store>100000): continue

        # step 4: Length of stay
        los_days = (visit.discharge_time - visit.encounter_time).days
        los_category = categorize_los(los_days)

        # step 4: assemble the samples
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                # the following keys can be the "feature_keys" or "label_key" for initializing downstream ML model
                "conditions": conditions,
                "procedures": procedures,
                "drugs": drugs,
                "label_mortality_class": mortality_label,
                "label_length_of_stay": los_category,
            }
        )
        if mortality_label == 0:
            tmp_store = 1 + tmp_store
    return samples

from pyhealth.data import Patient, Visit
def mortality_prediction_mimic4_fn(patient: Patient):
    """Processes a single patient for the mortality prediction task.

    Mortality prediction aims at predicting whether the patient will decease in the
    next hospital visit based on the clinical information from current visit
    (e.g., conditions and procedures).

    Args:
        patient: a Patient object

    Returns:
        samples: a list of samples, each sample is a dict with patient_id,
            visit_id, and other task-specific attributes as key

    Note that we define the task as a binary classification task.

    Examples:
        >>> from pyhealth.datasets import MIMIC4Dataset
        >>> mimic4_base = MIMIC4Dataset(
        ...     root="/srv/local/data/physionet.org/files/mimiciv/2.0/hosp",
        ...     tables=["diagnoses_icd", "procedures_icd"],
        ...     code_mapping={"ICD10PROC": "CCSPROC"},
        ... )
        >>> from pyhealth.tasks import mortality_prediction_mimic4_fn
        >>> mimic4_sample = mimic4_base.set_task(mortality_prediction_mimic4_fn)
        >>> mimic4_sample.samples[0]
        [{'visit_id': '130744', 'patient_id': '103', 'conditions': [['42', '109', '19', '122', '98', '663', '58', '51']], 'procedures': [['1']], 'label': 1}]
    """
    samples = []

    # we will drop the last visit
    for i in range(len(patient) - 1):
        visit: Visit = patient[i]
        next_visit: Visit = patient[i + 1]

        if next_visit.discharge_status not in [0, 1]:
            mortality_label = 0
        else:
            mortality_label = int(next_visit.discharge_status)

        conditions = visit.get_code_list(table="diagnoses_icd")
        procedures = visit.get_code_list(table="procedures_icd")
        drugs = visit.get_code_list(table="prescriptions")
        # exclude: visits without condition, procedure, or drug code
        if len(conditions) * len(procedures) * len(drugs) == 0:
            continue
        # TODO: should also exclude visit with age < 18
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": [conditions],
                "procedures": [procedures],
                "drugs": [drugs],
                "label": mortality_label,
            }
        )
    # no cohort selection
    return samples

def mortality_prediction_lenght_stay_mimic4(patient: Patient):

    samples = []
    global tmp_store
    # we will drop the last visit
    mortality_label = 0
    for i in range(len(patient) - 1):
        visit: Visit = patient[i]
        next_visit: Visit = patient[i + 1]

        if next_visit.discharge_status not in [0, 1]:
            mortality_label = 0
        else:
            mortality_label = int(next_visit.discharge_status)

        conditions = visit.get_code_list(table="diagnoses_icd")
        procedures = visit.get_code_list(table="procedures_icd")
        drugs = visit.get_code_list(table="prescriptions")
        #Length of stay
        los_days = (visit.discharge_time - visit.encounter_time).days
        los_category = categorize_los(los_days)
        # exclude: visits without condition, procedure, or drug code
        if len(conditions) * len(procedures) * len(drugs) == 0: continue
        if mortality_label == 0 and tmp_store > 4500: continue
        # TODO: should also exclude visit with age < 18
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": [conditions],
                "procedures": [procedures],
                "drugs": [drugs],
                "label_mortality_class": mortality_label,
                "label_length_of_stay": los_category,
            }
        )
    if mortality_label == 0:
        tmp_store = 1 + tmp_store
    # no cohort selection
    return samples

def aggregating_data(args):
    from pyhealth.datasets import MIMIC3Dataset
    if not args.dev: # if 0 we use dev set (subsample)
        dev=True
    else:
        dev = False #/home/e/e1100042/scratch/physionet.org/files/mimiciii/1.4 "/media/yannis/T7 Touch/1.4/"

    # Set the number of workers to a lower value
    if args.dataset == "mimic_3":
        mimic3_ds = MIMIC3Dataset(
            root=args.path_dataset,
            tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
            code_mapping={},
            dev=dev,
            refresh_cache=False,
        )
        # Load lenght of stay task + mortality prediction
        dataset = mimic3_ds.set_task(mortality_prediction_lenght_stay)
    else:
        from pyhealth.datasets import MIMIC4Dataset
        # "/media/yannis/T7 Touch/physionet.org/files/mimiciv/3.0/hosp/",
        mimic4_base = MIMIC4Dataset(
            root=args.path_dataset,
            tables=["diagnoses_icd", "procedures_icd", 'prescriptions'],
            code_mapping={"ICD10PROC": "CCSPROC"},
            dev=dev,
            refresh_cache=False
        )
        # Load lenght of stay task + mortality prediction
        # mimic4_sample = mimic4_base.set_task(mortality_prediction_mimic4_fn) #change
        dataset = mimic4_base.set_task(mortality_prediction_lenght_stay_mimic4)
        print("Dataset loaded")
        print("Number of patients: ", len(dataset))
        # check imbalance
        print("Percentage of patients with mortality: ", sum([1 for i in range(len(dataset)) if dataset.samples[i]
                                        ['label_mortality_class'] == 1])/len(dataset))

    from pyhealth.datasets import split_by_patient, get_dataloader

    # data split
    rng = np.random.RandomState(42)
    args.state = rng
    train_dataset, val_dataset, test_dataset = split_by_patient(dataset, [0.8, 0.2, 0.0], seed=42)
    # create dataloaders (they are <torch.data.DataLoader> object)
    train_loader_init = get_dataloader(train_dataset, batch_size=args.batch_size, shuffle=False)
    val_loader_init = get_dataloader(val_dataset, batch_size=args.batch_size, shuffle=False)

    expert_tr, expert_val = experts_sampling(args, args.device,{'train': train_loader_init, 'val': val_loader_init}, dataset, args.name)
    for j in range(len(train_loader_init.dataset.indices)):
        id = train_loader_init.dataset.indices[j]
        train_loader_init.dataset.dataset.samples[id]['expert_class'] = expert_tr['expert_class_tr'][j]
        train_loader_init.dataset.dataset.samples[id]['expert_reg'] = expert_tr['expert_reg_tr'][j]
    for j in range(len(val_loader_init.dataset.indices)):
        id = val_loader_init.dataset.indices[j]
        val_loader_init.dataset.dataset.samples[id]['expert_class'] = expert_val['expert_class_val'][j]
        val_loader_init.dataset.dataset.samples[id]['expert_reg'] = expert_val['expert_reg_val'][j]


    print("Number of patients in train: ", len(train_loader_init.dataset))
    print("Number of patients in val: ", len(val_loader_init.dataset))

    if args.overfit:
        train_loader_init = val_loader_init

    return {'train': train_loader_init, 'val': val_loader_init}, dataset

def preprocess_data(args):
    loader, dataset = aggregating_data(args)
    train_loader = loader['train']
    val_loader = loader['val']
    return {'train': train_loader, 'val': val_loader}, dataset

def expert_clusters(args, device, dataloader, dataset, name):
    dir_name = f"./logs/train/{args.dataset}/two_stage_{name}"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    from sklearn.cluster import KMeans
    classifier_init = Custom(
        dataset=dataset,
        feature_keys=["conditions", "procedures"],
        label_key='label_mortality_class',  # ignore this for now
        mode="multiclass",
    )
    classifier = nn.DataParallel(classifier_init.to(device))
    store = []
    patient_id = []
    label_id = []
    label_reg = []
    for i, x in enumerate(dataloader['train']):
        out = classifier(**x)
        t = torch.cat(out, dim=2).squeeze()
        store.append(t[:, :20])
        patient_id.append(x['patient_id'])
        label_id.extend(x['label_mortality_class'])
        label_reg.extend(x['label_length_of_stay'])
    data = torch.cat(store, dim=0).cpu().numpy()
    labels_reg = torch.tensor(label_reg).to(device)

    label_id_val = []
    label_reg_val = []
    patient_id_val = []
    store_val = []
    for i, x in enumerate(dataloader['val']):
        out = classifier(**x)
        t = torch.cat(out, dim=2).squeeze()
        if len(t.shape) ==1:
            t = t.unsqueeze(0)
        store_val.append(t[:, :20])
        patient_id_val.append(x['patient_id'])
        label_id_val.extend(x['label_mortality_class'])
        label_reg_val.extend(x['label_length_of_stay'])
    data_val = torch.cat(store_val, dim=0).cpu().numpy()
    labels_reg_val = torch.tensor(label_reg_val).to(device)
    data_kmeans = np.concatenate((data, data_val), axis=0)
    # Perform KMeans clustering to find 6 clusters
    N = 6
    # Initialize the KMeans model
    kmeans = KMeans(
        n_clusters=N,
        init='k-means++',
        max_iter=1000,
        tol=1e-6,
        random_state=42
    )
    kmeans.fit(data_kmeans)
    global_cluster = kmeans.labels_

    cluster_labels = kmeans.predict(data)

    exp1_correctness = np.logical_or(cluster_labels == 0, cluster_labels == 1, cluster_labels == 3)
    exp2_correctness = np.logical_or(cluster_labels == 0, cluster_labels == 4, cluster_labels == 5)

    expert1_cla = np.where(exp1_correctness, np.asarray(label_id), np.abs(np.asarray(label_id)-1))[:, None]
    expert2_cla = np.where(exp2_correctness, np.asarray(label_id), np.abs(np.asarray(label_id)-1))[:, None]
    expert_class_tr = torch.tensor(np.concatenate((expert1_cla, expert2_cla), axis=1)).to(args.device)

    expert1_reg = np.where(exp1_correctness, labels_reg.cpu().numpy(),
                           args.state.choice(labels_reg.cpu().numpy(), size=cluster_labels.shape))[:, None]
    expert2_reg = np.where(exp2_correctness, labels_reg.cpu().numpy(),
                           args.state.choice(labels_reg.cpu().numpy(), size=cluster_labels.shape))[:, None]
    expert_reg_tr = torch.tensor(np.concatenate((expert1_reg, expert2_reg), axis=1)).to(args.device)

    store_cluster = cluster_labels

    # Validation
    cluster_labels = kmeans.predict(data_val)

    exp1_correctness = np.logical_or(cluster_labels == 0, cluster_labels == 1, cluster_labels == 3)
    exp2_correctness = np.logical_or(cluster_labels == 0, cluster_labels == 4, cluster_labels == 5)

    expert1_cla = np.where(exp1_correctness, np.asarray(label_id_val), np.abs(np.asarray(label_id_val)-1))[:, None]
    expert2_cla = np.where(exp2_correctness, np.asarray(label_id_val), np.abs(np.asarray(label_id_val)-1))[:, None]
    expert_class_val = torch.tensor(np.concatenate((expert1_cla, expert2_cla), axis=1)).to(args.device)

    expert1_reg = np.where(exp1_correctness, labels_reg_val.cpu().numpy(),
                           labels_reg_val.cpu().numpy() + 1 + args.state.choice(labels_reg_val.cpu().numpy(), size=cluster_labels.shape))[:, None]
    expert2_reg = np.where(exp2_correctness, labels_reg_val.cpu().numpy(),
                           labels_reg_val.cpu().numpy() + 1 + args.state.choice(labels_reg_val.cpu().numpy(), size=cluster_labels.shape))[:, None]
    expert_reg_val = torch.tensor(np.concatenate((expert1_reg, expert2_reg), axis=1)).to(args.device)

    print(f'accuracy expert 1 validation: {np.mean(expert1_cla[:,0] == np.asarray(label_id_val))}')
    print(f'accuracy expert 2 validation: {np.mean(expert2_cla[:,0] == np.asarray(label_id_val))}')

    # Save the cluster centers
    import sys
    script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    path = f'/dataset/experts/{args.dataset}'
    dir = script_dir + path
    import pickle
    with open(dir + '_kmeans_model.pkl', 'wb') as file:
        pickle.dump(kmeans, file)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    plt.hist(store_cluster, bins=N, edgecolor='black', alpha=0.7)
    plt.title('Distribution of Cluster Labels')
    plt.xlabel('Cluster Label')
    plt.ylabel('Number of Points')
    plt.xticks(range(N))  # Since we have 4 clusters
    plt.savefig(dir + 'cluster_distribution_training.png')

    # Save the cluster centers
    plt.figure(figsize=(8, 6))
    plt.hist(cluster_labels, bins=N, edgecolor='black', alpha=0.7)
    plt.title('Distribution of Cluster Labels')
    plt.xlabel('Cluster Label')
    plt.ylabel('Number of Points')
    plt.xticks(range(N))  # Since we have 4 clusters
    plt.savefig(dir + 'cluster_distribution_validation.png')

    # Save the cluster centers
    plt.figure(figsize=(8, 6))
    plt.hist(global_cluster, bins=N, edgecolor='black', alpha=0.7)
    plt.title('Distribution of Cluster Labels')
    plt.xlabel('Cluster Label')
    plt.ylabel('Number of Points')
    plt.xticks(range(N))  # Since we have 4 clusters
    plt.savefig(dir + 'cluster_distribution_global.png')

    return ({'expert_class_tr': expert_class_tr.tolist(), 'expert_reg_tr': expert_reg_tr.tolist()},
            {'expert_class_val': expert_class_val.tolist(), 'expert_reg_val': expert_reg_val.tolist()})

def expert_oracle(args, device, dataloader, dataset, name):
    label_id = []
    label_reg = []
    for i, x in enumerate(dataloader['train']):
        label_id.extend(x['label_mortality_class'])
        label_reg.extend(x['label_length_of_stay'])
    expert_reg_tr = torch.tensor(label_reg).to(device)
    expert_class_tr = torch.tensor(label_id).to(device)
    expert_tr = {'expert_class_tr': expert_class_tr.tolist(), 'expert_reg_tr': expert_reg_tr.tolist()}

    label_id = []
    label_reg = []
    for i, x in enumerate(dataloader['val']):
        label_id.extend(x['label_mortality_class'])
        label_reg.extend(x['label_length_of_stay'])
    expert_reg_val = torch.tensor(label_reg).to(device)
    expert_class_val = torch.tensor(label_id).to(device)
    expert_val = {'expert_class_val': expert_class_val.tolist(), 'expert_reg_val': expert_reg_val.tolist()}
    return expert_tr, expert_val

def experts_sampling(args, device, dataloader, dataset, name):
    if args.expert_exp == 'Oracle':
        expert_tr, expert_val = expert_oracle(args, device, dataloader, dataset, name)

    elif args.expert_exp == 'Clusters':
        expert_tr, expert_val = expert_clusters(args, device, dataloader, dataset, name)

    else:
        raise NotImplementedError
    
    return expert_tr, expert_val












