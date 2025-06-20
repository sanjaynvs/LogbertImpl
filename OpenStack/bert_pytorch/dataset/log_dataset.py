from torch.utils.data import Dataset
import torch
import random
import numpy as np
from collections import defaultdict

class LogDataset(Dataset):
    def __init__(self, log_corpus, time_corpus, vocab, seq_len, corpus_lines=None, encoding="utf-8", on_memory=True, predict_mode=False, mask_ratio=0.15):
        """

        :param corpus: log sessions/line
        :param vocab: log events collection including pad, ukn ...
        :param seq_len: max sequence length
        :param corpus_lines: number of log sessions
        :param encoding:
        :param on_memory:
        :param predict_mode: if predict
        """
        # print("LogDataset:__init__")
        #print(log_corpus[:100]) -- verfied
        # print("log_corpus size: ", len(log_corpus))
        self.vocab = vocab
        self.seq_len = seq_len

        self.on_memory = on_memory
        self.encoding = encoding

        self.predict_mode = predict_mode
        self.log_corpus = log_corpus
        self.time_corpus = time_corpus
        self.corpus_lines = len(log_corpus)

        self.mask_ratio = mask_ratio

    def __len__(self):
        # print("LogDataset:__len__")
        return self.corpus_lines

    def __getitem__(self, idx):
        print("LogDataset:__getitem__", idx)
        k, t = self.log_corpus[idx], self.time_corpus[idx]

        k_masked, k_label, t_masked, t_label = self.random_item(k, t)

        # [CLS] tag = SOS tag, [SEP] tag = EOS tag
        k = [self.vocab.sos_index] + k_masked
        k_label = [self.vocab.pad_index] + k_label
        # k_label = [self.vocab.sos_index] + k_label

        t = [0] + t_masked
        t_label = [self.vocab.pad_index] + t_label

        return k, k_label, t, t_label

    def random_item(self, k, t):
        print("LogDataset:random_item...start")
        tokens = list(k)
        print("tokens", tokens)
        output_label = []

        time_intervals = list(t)
        time_label = []

        for i, token in enumerate(tokens):
            print("start of for....")
            print("tokens", tokens)
            print("i", i)
            print("token", token)
            time_int = time_intervals[i]
            prob = random.random()
            print("prob", prob)
            # print("self.mask_ratio", self.mask_ratio)
            # replace 15% of tokens in a sequence to a masked token
            if prob < self.mask_ratio:
                # raise AttributeError(
                # "no mask in visualization")
                print("if # 1")
                # print("self.predict_mode",self.predict_mode)
                if self.predict_mode:
                    print("line after if self.predict_mode...")
                    # print(" before assignment value ooff  tokens[i]",tokens[i])
                    # print("self.vocab.mask_index",self.vocab.mask_index)
                    tokens[i] = self.vocab.mask_index
                    print("\ttokens",tokens)
                    # print("\tself.vocab.unk_index",self.vocab.unk_index)
                    # print("\t#1 self.vocab.stoi.get(token, self.vocab.unk_index)",self.vocab.stoi.get(token, self.vocab.unk_index))
                    output_label.append(self.vocab.stoi.get(token, self.vocab.unk_index))
                    print("\t#1 output_label",output_label)

                    time_label.append(time_int)
                    time_intervals[i] = 0
                    continue

                print("prob #1",prob)    
                prob /= self.mask_ratio
                print("prob # 2", prob)
                
                # 80% randomly change token to mask token
                if prob < 0.8:
                    # print("if # 2")
                    # print("\tself.vocab.mask_index",self.vocab.mask_index)
                    tokens[i] = self.vocab.mask_index

                # 10% randomly change token to random token
                elif prob < 0.9:
                    # print("if # 3")
                    # print("\trandom.randrange(len(self.vocab))",random.randrange(len(self.vocab)))
                    tokens[i] = random.randrange(len(self.vocab))

                # 10% randomly change token to current token
                else:
                    # print("else #1")
                    # print("\tself.vocab.stoi.get(token, self.vocab.unk_index)",self.vocab.stoi.get(token, self.vocab.unk_index))
                    tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)
                print("\t#2 self.vocab.stoi.get(token, self.vocab.unk_index)",self.vocab.stoi.get(token, self.vocab.unk_index))
                output_label.append(self.vocab.stoi.get(token, self.vocab.unk_index))

                time_intervals[i] = 0  # time mask value = 0
                time_label.append(time_int)

            else:
                print("else #2")
                # print("\t#3 self.vocab.stoi.get(token, self.vocab.unk_index)",self.vocab.stoi.get(token, self.vocab.unk_index))
                tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)
                output_label.append(0)
                print("\ttokens",tokens)
                print("\output_label",output_label)
                time_label.append(0)

        return tokens, output_label, time_intervals, time_label

    def collate_fn(self, batch, percentile=100, dynamical_pad=True):
        # print("in collate_fn", len(batch))
        # print(batch)
        lens = [len(seq[0]) for seq in batch]

        # find the max len in each batch
        if dynamical_pad:
            # dynamical padding
            seq_len = int(np.percentile(lens, percentile))
            if self.seq_len is not None:
                seq_len = min(seq_len, self.seq_len)
        else:
            # fixed length padding
            seq_len = self.seq_len

        output = defaultdict(list)
        for seq in batch:
            bert_input = seq[0][:seq_len]
            bert_label = seq[1][:seq_len]
            time_input = seq[2][:seq_len]
            time_label = seq[3][:seq_len]

            padding = [self.vocab.pad_index for _ in range(seq_len - len(bert_input))]
            bert_input.extend(padding), bert_label.extend(padding), time_input.extend(padding), time_label.extend(
                padding)

            time_input = np.array(time_input)[:, np.newaxis]
            output["bert_input"].append(bert_input)
            output["bert_label"].append(bert_label)
            output["time_input"].append(time_input)
            output["time_label"].append(time_label)

        # output["bert_input"] = torch.tensor(output["bert_input"], dtype=torch.long)
        output["bert_input"] = torch.tensor(np.array(output["bert_input"]), dtype=torch.long)
        output["bert_label"] = torch.tensor(np.array(output["bert_label"]), dtype=torch.long)
        output["time_input"] = torch.tensor(np.array(output["time_input"]), dtype=torch.float)
        output["time_label"] = torch.tensor(np.array(output["time_label"]), dtype=torch.float)
        # print("="*40)
        # print("output[bert_input]")
        # print(output["bert_input"])

        # print("="*40)
        # print("output[bert_label]")
        # print(output["bert_label"])
       
        return output
