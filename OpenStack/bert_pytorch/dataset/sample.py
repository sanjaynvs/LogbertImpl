from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split


def generate_pairs(line, window_size):
    line = np.array(line)
    line = line[:, 0]

    seqs = []
    for i in range(0, len(line), window_size):
        seq = line[i:i + window_size]
        seqs.append(seq)
    seqs += []
    seq_pairs = []
    for i in range(1, len(seqs)):
        seq_pairs.append([seqs[i - 1], seqs[i]])
    return seqs


def fixed_window(line, window_size, adaptive_window, seq_len=None, min_len=0):
    # print("in fixed_window")
    # print("line", line)
    line = [ln.split(",") for ln in line.split()]
    # print("line after split", line)

    # filter the line/session shorter than 10
    # print(len(line), "min_len", min_len)
    if len(line) < min_len:
        # print("session too short, skip")
        return [], []

    # max seq len
    # print("seq_len", seq_len)
    if seq_len is not None:
        line = line[:seq_len]
        # print("after seq_len", line)

    if adaptive_window:
        window_size = len(line)
        # print("adaptive_window, window_size", window_size)

    line = np.array(line)
    # print("line after np.array", line)
    # print("line shape", line.shape[1])

    # if time duration exists in data
    if line.shape[1] == 2:
        tim = line[:,1].astype(float)
        line = line[:, 0]

        # the first time duration of a session should be 0, so max is window_size(mins) * 60
        tim[0] = 0
       
    else:
        line = line.squeeze()
        line = line.astype(int)
        # print("line shape after squeeze", line.shape)
        # print("line", line) 
        # print("line type", type(line))
        # if time duration doesn't exist, then create a zero array for time
        tim = np.zeros(line.shape)

    logkey_seqs = []
    time_seq = []
    for i in range(0, len(line), window_size):
        # print("going to append seqs", i, i + window_size, "len(line)", len(line))
        logkey_seqs.append(line[i:i + window_size])
        time_seq.append(tim[i:i + window_size])
        

    return logkey_seqs, time_seq


def generate_train_valid(data_path, window_size=20, adaptive_window=True,
                         sample_ratio=1, valid_size=0.1, output_path=None,
                         scale=None, scale_path=None, seq_len=None, min_len=0):
    with open(data_path, 'r') as f:
        data_iter = f.readlines()

    num_session = int(len(data_iter) * sample_ratio)
    # only even number of samples, or drop_last=True in DataLoader API
    # coz in parallel computing in CUDA, odd number of samples reports issue when merging the result
    # num_session += num_session % 2

    test_size = int(min(num_session, len(data_iter)) * valid_size)
    # only even number of samples
    # test_size += test_size % 2

    print("before filtering short session")
    print("train size ", int(num_session - test_size))
    print("valid size ", int(test_size))
    print("="*40)

    logkey_seq_pairs = []
    time_seq_pairs = []
    session = 0
    counter=0
    for line in tqdm(data_iter):
        counter += 1
        if session >= num_session:
            break
        session += 1

        logkeys, times = fixed_window(line, window_size, adaptive_window, seq_len, min_len)
        # print("before...logkey_seq_pairs", logkey_seq_pairs[:5])
        # print("logkeys", logkeys)
        # print("times", times)
        logkey_seq_pairs += logkeys
        time_seq_pairs += times
        # print("logkey_seq_pairs",0 len(logkey_seq_pairs))
        # print("time_seq_pairs", len(time_seq_pairs))
    
    print("counter", counter)
    print("num of sessions", session)
    print("num of seqs", len(logkey_seq_pairs))
    print("num of time seqs", len(time_seq_pairs))

    # print("dir(logkey_seq_pairs)", dir(logkey_seq_pairs))
    # print("type(logkey_seq_pairs)", type(logkey_seq_pairs))
    # for element in logkey_seq_pairs:
    #     print("element", element)
    #     print("type(element)", type(element))
    #     print("len(element)", len(element))

    logkey_seq_pairs = np.array(logkey_seq_pairs, dtype=object)
    time_seq_pairs = np.array(time_seq_pairs, dtype=object)

    print("num of logkey seqs", len(logkey_seq_pairs))
    print("num of time seqs", len(time_seq_pairs))

    logkey_trainset, logkey_validset, time_trainset, time_validset = train_test_split(logkey_seq_pairs,
                                                                                      time_seq_pairs,
                                                                                      test_size=test_size,
                                                                                      random_state=1234)

    # sort seq_pairs by seq len
    train_len = list(map(len, logkey_trainset))
    # print("train_len", train_len)
    valid_len = list(map(len, logkey_validset))
    # print("valid_len", valid_len)

    train_sort_index = np.argsort(-1 * np.array(train_len))
    # print("train_sort_index", train_sort_index)
    valid_sort_index = np.argsort(-1 * np.array(valid_len))
    # print("valid_sort_index", valid_sort_index)

    logkey_trainset = logkey_trainset[train_sort_index]
    logkey_validset = logkey_validset[valid_sort_index]

    time_trainset = time_trainset[train_sort_index]
    time_validset = time_validset[valid_sort_index]

    print("="*40)
    print("Num of train seqs", len(logkey_trainset))
    print("Num of valid seqs", len(logkey_validset))
    print("="*40)

    return logkey_trainset, logkey_validset, time_trainset, time_validset
