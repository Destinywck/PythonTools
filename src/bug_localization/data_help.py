#! /usr/bin/env python
import pandas as pd
import numpy as np
import re


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def pad_sentences_dict(sentences, sequence_length):
    padded_sentences = {}
    for i, j in sentences.items():
        sentence = str(j).split(' ')
        if len(sentence) < sequence_length:
            new_sentence = str(j)
        else:
            new_sentence = ' '.join(sentence[:sequence_length])
        padded_sentences[i] = new_sentence
    return padded_sentences


def load_data_csv(params, bug_text_file, code_text_file, bug_code_index_file):
    # Load data from file
    selected = ['bugText', 'codePath', 'codeText', 'label', 'bugId']

    bug_text_df = pd.read_csv(bug_text_file)
    code_text_df = pd.read_csv(code_text_file)
    bug_code_index_df = pd.read_csv(bug_code_index_file)

    # bug text
    bug_text_index = bug_text_df[selected[4]].tolist()
    bug_text_raw = bug_text_df[selected[0]].apply(lambda x: clean_str(x)).tolist()
    bug_text = {}
    for i in range(len(bug_text_index)):
        bug_text[str(bug_text_index[i])] = bug_text_raw[i]

    # code text
    code_text_index = code_text_df[selected[1]].tolist()
    code_text_raw = code_text_df[selected[2]].tolist()
    code_text = {}
    for i in range(len(code_text_index)):
        code_text[code_text_index[i]] = code_text_raw[i]

    bug_text = pad_sentences_dict(bug_text, params["max_len_left"])
    code_text = pad_sentences_dict(code_text, params["max_len_right"])

    # count = 0
    # for i, j in bug_text.items():
    #     print(i + ' ' + j)
    #     count += 1
    #     if count > 10:
    #         break
    #
    # count = 0
    # for i, j in code_text.items():
    #     print(i + ' ' + j)
    #     count += 1
    #     if count > 10:
    #         break

    data_left = []
    data_right = []
    data_label = []
    bug_index = bug_code_index_df[selected[4]].tolist()
    bug_codes = bug_code_index_df[selected[1]].tolist()
    bug_code_label = bug_code_index_df[selected[3]].tolist()
    bug_code_label = [int(l) for l in bug_code_label]
    for i in range(len(bug_index)):
        for path in str(bug_codes[i]).split('#'):
            data_left.append(bug_text[str(bug_index[i])])
            data_right.append(code_text[path])
            data_label.append(bug_code_label[i])

    num_pos = sum(data_label)
    data_label = [[0, 1] if i == 1 else [1, 0] for i in data_label]
    data_label = np.asanyarray(data_label)

    # print(data_left, data_right, data_label)
    return data_left, data_right, data_label, num_pos


def load_text_and_label(bug_text_file, code_text_file, bug_code_index_file):
    """

    :param bug_text_file:
    :param code_text_file:
    :param bug_code_index_file:
    :return:
    """

    # Load data from file
    selected = ['bugText', 'codePath', 'codeText', 'label', 'bugId']

    bug_text_df = pd.read_csv(bug_text_file)
    code_text_df = pd.read_csv(code_text_file)
    bug_code_index_df = pd.read_csv(bug_code_index_file)

    # bug text
    bug_text_index = bug_text_df[selected[4]].tolist()
    bug_text_raw = bug_text_df[selected[0]].apply(lambda x: clean_str(x)).tolist()
    bug_text = {}
    for i in range(len(bug_text_index)):
        bug_text[bug_text_index[i]] = bug_text_raw[i]

    # code text
    code_text_index = code_text_df[selected[1]].tolist()
    code_text_raw = code_text_df[selected[2]].tolist()
    code_text = {}
    for i in range(len(code_text_index)):
        code_text[code_text_index[i]] = code_text_raw[i]

    x_input = []
    y_input = []
    y_label = []
    bug_index = bug_code_index_df[selected[4]].tolist()
    bug_codes = bug_code_index_df[selected[1]].tolist()
    bug_code_label = bug_code_index_df[selected[3]].tolist()
    for i in range(len(bug_index)):
        for path in bug_codes[i].split('#'):
            x_input.append(bug_text[bug_index[i]])
            y_input.append(code_text[path])
            y_label.append(bug_code_label[i])

    y_label = [[0, 1] if i == 1 else [1, 0] for i in y_label]

    y_label = np.asanyarray(y_label)
    print('data set size: ' + str(len(y_label)))
    # print(x_input[0 : 10])
    # print(y_input[0 : 10])
    # print(y_label)
    return [x_input, y_input, y_label]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]