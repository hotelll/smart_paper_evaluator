# -*- coding:utf-8 -*-

import os
import fitz
import re
import random
import csv


INPUT_DIR = '../input/CVPR/'
OUTPUT_DIR = '../output/'
TEST_SIZE = 0.2

TEXT_BLOCKSIZE = 2000


def get_paper_seq(file_path):
    doc = fitz.open(file_path)

    text = ""

    page_num = doc.page_count

    for pid in range(page_num):
        page = doc[pid]
        text = text + page.get_text()

    text_length = len(text)

    fig_iter = re.finditer(r'\n(Figure\s\d+\.)', text)
    figs = []
    for f in fig_iter:
        figs.append([f.span()[0], f.span()[1]])

    tab_iter = re.finditer(r'\n(Table\s\d+\.)', text)
    tabs = []
    for t in tab_iter:
        tabs.append([t.span()[0], t.span()[1]])

    temp = ["W"] * text_length

    for area in figs:
        temp[area[0]] = "F"
    for area in tabs:
        temp[area[0]] = "T"

    paper_seq = ""
    i = 0
    while i < text_length:
        pointer = 0
        for j in range(min(TEXT_BLOCKSIZE, text_length-i)):
            pointer = j+1
            if temp[i + j] == "F":
                paper_seq = paper_seq + "F "
                break
            elif temp[i + j] == "T":
                paper_seq = paper_seq + "T "
                break
        if pointer == TEXT_BLOCKSIZE:   # no figure or table
            paper_seq = paper_seq + "W "
        i += pointer
    return paper_seq


if __name__ == '__main__':
    print('execute seq_process.py ...')
    # train_test_split
    pos_lst = os.listdir('../input/CVPR/raw/positive')
    neg_lst = os.listdir('../input/CVPR/raw/negative')

    paper_paths = []
    for i in pos_lst:
        paper_paths.append(["1", '../input/CVPR/raw/positive/' + i])
    for i in neg_lst:
        paper_paths.append(["0", '../input/CVPR/raw/negative/' + i])

    random.shuffle(paper_paths)
    index = 1
    with open("../input/seq_output/paper_seq.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["label", "content"])
        for label, path in paper_paths:
            seq = ""
            try:
                seq = get_paper_seq(path)
            except:
                continue
            if seq == "":
                seq = "W"
            print(index)
            index += 1
            writer.writerow([label, seq])
