import sys
sys.path.append('../')

import os
import re
import json
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from logparser import Spell, Drain

# get [log key, delta time] as input for deeplog
input_dir  = './datasets'
output_dir = './output/openstack/'  # The output directory of parsing results
#log_file   = "HDFS_2K.log"  # The input log file name
# log_file   = "OpenStack_2k.log"
# log_file   = "OpenStack_mini.log"
# log_file   = "OpenStack_abnormal.log"
# log_file   = "merged_logs.log"
log_file   = "common2.log"
log_structured_file = output_dir + log_file + "_structured.csv"
# log_structured_file = output_dir + "common2_structured.csv"
log_templates_file = output_dir + log_file + "_templates.csv"
log_sequence_file = output_dir + "openstack_sequence.csv"

def mapping():
    log_temp = pd.read_csv(log_templates_file)
    log_temp.sort_values(by = ["Occurrences"], ascending=False, inplace=True)
    log_temp_dict = {event: idx+1 for idx , event in enumerate(list(log_temp["EventId"])) }
    # print(log_temp_dict)
    with open (output_dir + "openstack_log_templates.json", "w") as f:
        json.dump(log_temp_dict, f)


def parser(input_dir, output_dir, log_file, log_format, type='drain'):
    if type == 'spell':
        tau        = 0.5  # Message type threshold (default: 0.5)
        regex      = [
            r"(/[-\w]+)+", #replace file path with *
            r"(?<=blk_)[-\d]+" #replace block_id with *

        ]  # Regular expression list for optional preprocessing (default: [])

        parser = Spell.LogParser(indir=input_dir, outdir=output_dir, log_format=log_format, tau=tau, rex=regex, keep_para=False)
        parser.parse(log_file)

    elif type == 'drain':
        regex = [
            #r"[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}", #compute instance id e.g. a208479c-c0e3-4730-a5a0-d75e8afd0252
            #r'^[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12},\d+,[a-fA-F0-9]+,[a-fA-F0-9]{32} [a-fA-F0-9]{32} (?:- ){3}-\]$',  # represents - 4b0c1f2e-3d5a-4c8b-9f6d-7e8f9a0b1c2d,34,462cb051,4b0c1f2e3d5a4c8b9f6d7e8f9a0b1c2d - - - -]'
            # in Creating event network-vif-plugged:a208479c-c0e3-4730-a5a0-d75e8afd0252 for instance 96abccce-8d1f-4e07-b6d1-4b2ab87e23b4,111,2da4176f,f7b8d1f1d4d44643b07fa10ca7d021fb e9746973ac574c6b8a9e8857f56a7608 - - -] 
			r'^[a-z0-9.-]+,\d+,[a-fA-F0-9]+,(?:- ){4}-\]$', # represents - cp-1.slowvm1.tcloud-pg0.utah.cloudlab.us,34,462cb051,- - - - -] 
            r'[a-z0-9.-]+(?:\.[a-z0-9-]+)+',
            r"[a-f0-9]{8}-[a-f0-9]{4}-[1-5][a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}",  #compute instance id e.g. a208479c-c0e3-4730-a5a0-d75e8afd0252
            r"(/[-\w]+)+",  # file path
            r'\d+\.\d+\.\d+\.\d+',  # IP
            r"(/[-\w]+)+ HTTP/1.1",  # file path with HTTP protocol
            r'\d+(\.\d+)?',  # Numbers with optional decimal point
        ]

        # the hyper parameter is set according to http://jmzhu.logpai.com/pub/pjhe_icws2017.pdf
        st = 0.5  # Similarity threshold
        depth = 10  # Depth of all leaf nodes

        parser = Drain.LogParser(log_format, indir=input_dir, outdir=output_dir, depth=depth, st=st, rex=regex, keep_para=False)
        parser.parse(log_file)


def openstack_sampling(log_file, window='session'):
    assert window == 'session', "Only window=session is supported for OS dataset."
    print("Loading", log_file)
    df = pd.read_csv(log_file, engine='c',
            na_filter=False, memory_map=True, dtype={'Date':object, "Time": object})

    with open(output_dir + "openstack_log_templates.json", "r") as f:
        event_num = json.load(f)
    df["EventId"] = df["EventId"].apply(lambda x: event_num.get(x, -1))
    
    data_dict = defaultdict(list) #preserve insertion order of items
    for idx,row in tqdm(df.iterrows()):
        # if row['Component']!= 'nova.metadata.wsgi.server' :
        computeInst_list = re.findall(r'[a-f0-9]{8}-[a-f0-9]{4}-[1-5][a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}', row['Content'])
        computeInst_set = set(computeInst_list)
        for computeInst_Id in computeInst_set:
            data_dict[computeInst_Id].append(row["EventId"])
                #for dataanalysis...
                #data_dict[computeInst_Id].append(row['E_id'])

    data_df = pd.DataFrame(list(data_dict.items()), columns=['ComputeInstance', 'EventSequence'])
    # print(data_df.head())
    data_df.to_csv(log_sequence_file, index=None)
    print("OS sampling done")

def _custom_resampler(array_like):
    return list(array_like)

def generate_train_test(compute_instance_file, n=None, ratio=0.3):
    # print("Loading", compute_instance_file)
    computeInst_label_dict = {}
    computeInst_label_file = os.path.join(input_dir, "anomaly_label.csv")
    computeInst_df = pd.read_csv(computeInst_label_file)
    for idx, row in tqdm(computeInst_df.iterrows()):
        computeInst_label_dict[row["ComputeInstance"]] = 1 if row["Label"] == "Anomaly" else 0

    # print(computeInst_label_dict)
    seq = pd.read_csv(compute_instance_file)
    # print(computeInst_label_dict.head())
    seq["Label"] = seq["ComputeInstance"].apply(lambda x: computeInst_label_dict.get(x)) #add label to the sequence of each ComputeInstance
    # print("seq[Label].value_counts()")
    # print(seq["Label"].value_counts())
    normal_seq = seq[seq["Label"] == 0]["EventSequence"]
    normal_seq = normal_seq.sample(frac=1, random_state=20) # shuffle normal data

    abnormal_seq = seq[seq["Label"] == 1]["EventSequence"]
    normal_len, abnormal_len = len(normal_seq), len(abnormal_seq)
    train_len = n if n else int(normal_len * ratio)
    print("normal size {0}, abnormal size {1}, training size {2}".format(normal_len, abnormal_len, train_len))

    train = normal_seq.iloc[:train_len]
    test_normal = normal_seq.iloc[train_len:]
    test_abnormal = abnormal_seq

    df_to_file(train, output_dir + "train")
    df_to_file(test_normal, output_dir + "test_normal")
    df_to_file(test_abnormal, output_dir + "test_abnormal")
    print("generate train test data done")


def df_to_file(df, file_name):
    with open(file_name, 'w') as f:
        for _, row in df.items():
            f.write(' '.join([str(ele) for ele in eval(row)]))
            f.write('\n')


if __name__ == "__main__":
    # 1. OS log
    # Below two lines are for parsing the log file. Will have to be refactored later 
    # log_format = '<Logrecord> <Date> <Time> <Pid> <Level> <Component> <ADDR> <Content>'  # OS log format
    log_format = '<Level> <Component> <ADDR> <Content>'  # OS log format
    parser(input_dir, output_dir, log_file, log_format, 'drain')
    mapping()
    openstack_sampling(log_structured_file)
    # generate_train_test(log_sequence_file, n=4855)
    # openstack_sampling("common2_strcutured.csv")
    generate_train_test(log_sequence_file)