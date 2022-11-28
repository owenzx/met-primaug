import argparse
import os
import csv
import random
from tqdm import tqdm
import json
import string
punct = string.punctuation

def load_scan_file(data_file):
    # Load SCAN dataset from file
    fid = open(data_file, 'r')
    lines = fid.readlines()
    fid.close()
    lines = [l.strip() for l in lines]
    lines = [l.lstrip('IN: ') for l in lines]
    src_tgt = [l.split(' OUT: ') for l in lines]
    commands = [{'src': l[0], 'trg': l[1]} for l in src_tgt]
    return commands

def load_cogs_data(data_path):
    all_data = []

    tsv_file = open(data_path)
    read_tsv = csv.reader(tsv_file, delimiter="\t")

    for row in read_tsv:
        src, trg, type = row
        all_data.append({
            'src': src,
            'trg': trg,
            'type': type
        })
    return all_data

def main(args):
    data = None
    if args.dataset_name == "COGS":
        args.data_dir = os.path.join(args.data_dir, 'cogs')
        data_file = os.path.join(args.data_dir, 'train.tsv')
        # with open(args.data_file, "r") as f:
        #     data = f.read().splitlines()
        data = load_cogs_data(data_file)
    elif args.dataset_name == "SCAN":
        if args.split == "addprim_jump":
            args.data_dir = os.path.join(args.data_dir, 'add_prim_split')
            data_file = os.path.join(args.data_dir, 'tasks_train_addprim_jump.txt')
            data = load_scan_file(data_file)
        if args.split == "around_right":
            args.data_dir = os.path.join(args.data_dir, 'template_split')
            data_file = os.path.join(args.data_dir, 'tasks_train_template_around_right.txt')
            data = load_scan_file(data_file)
        args.aligner_file = args.aligner_file + '_' + args.split

    word_alignment = {}

    inputs  = []
    outputs = []
    #with open(aligner_file, "r") as f:
    for k, line in enumerate(data):
        input, output = line['src'], line['trg']
        input, output = set(input.strip().split(" ")), set(output.strip().split(" "))
        inputs.append(input)
        outputs.append(output)
        for inp in input:
            if inp in punct:
                continue
            if inp not in word_alignment:
                word_alignment[inp] = {}
            inpmap = word_alignment[inp]
            for out in output:
                if out not in inpmap:
                    inpmap[out] = 1
                else:
                    inpmap[out] = inpmap[out] + 1
    # print(word_alignment['Emma'])
    for i in range(len(inputs)):
        input = inputs[i]
        output = outputs[i]
        for k in input:
            if k in punct:
                continue
            for v in list(word_alignment[k].keys()):
                if v not in output:
                    del word_alignment[k][v]  ### Remove trg words that are not suff(k,v) ###
            # else:
            #     for v in list(word_alignment[k].keys()):
            #         if v in outputs[i]:
            #             del word_alignment[k][v]
    # print(word_alignment['Emma'])
    ### Build the incoming dict ###
    incoming = {}
    for (k,mapped) in list(word_alignment.items()):
        for (v,_) in mapped.items():
            if v in incoming:
                incoming[v].add(k)
            else:
                incoming[v] = {k,}

        # if len(word_alignment[k]) == 0:
        #     del word_alignment[k]
    # print('incoming:')
    # print(incoming["Emma"])

    ### Remove (k,v) s.t. v has more than EPS different matching k ###
    for (v, inset) in incoming.items():
        if len(inset) > args.eps:
            # print(f"common word: v: {v}, inset: {inset}")
            # print("deleting ", v)
            for (k,mapped) in list(word_alignment.items()):
                if v in mapped and v != k:
                    #print(f"since EPS deleting {k}->{v}")
                    del word_alignment[k][v]
    # print(word_alignment['Emma'])

    for (v,inset) in incoming.items():
        if len(inset) > 1:
            candidates = set([e for e in inset])
            for k, line in enumerate(data):
                if len(candidates) == 0:
                    break
                input, output = line['src'], line['trg']
                input, output = set(input.strip().split(" ")), set(output.strip().split(" "))
                if v in output:
                    for e in set(candidates):
                        if e not in input:
                            candidates.remove(e)
            if len(candidates) == 1:  ### If we found there exist one candidate that suffice ness(t,v), then we remove all other wrong candidates
                wrongs = inset-candidates
                for t in wrongs:
                    if v in word_alignment[t]:
                        if t != v:
                            #print(f"in candidates deleting {t}->{v}")
                            del word_alignment[t][v]

    for (k,mapped) in list(word_alignment.items()):
        if len(word_alignment[k]) == 0:
            del word_alignment[k]
        else:
            if len(mapped) > 1:
                print(k, mapped)
            if k in mapped:
                mapped[k] += 1

            if len(mapped) > 1 and k in mapped:
                candidates = set([e for e in mapped])
                wrongs = candidates - set([k])
                for t in wrongs:
                    del word_alignment[k][t]
                print(k, mapped)


    # with open(aligner_file + '.v3.pickle', 'wb') as handle:
    #     pickle.dump(word_alignment, handle)
    src_word_count = {}
    for (k,mapped) in list(word_alignment.items()):
        src_word_count[k] = 0
        for dp in data:
            if k in dp['src']:
                src_word_count[k] += 1

    for (k, mapped) in list(word_alignment.items()):
        if src_word_count[k] > len(data) * args.threshold:
            del word_alignment[k]

    with open(os.path.join(args.data_dir, args.aligner_file + '.json'), 'w') as handle:
        json.dump(word_alignment, handle)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        type=str,
        default="addprim_jump",
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="../comp-data",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="COGS",
        choices=["COGS", "SCAN"]
    )
    parser.add_argument(
        "--aligner_file",
        type=str,
        default="aligned_vocab",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5
    )
    parser.add_argument(
        "--eps",
        type=int,
        default=3
    )
    args = parser.parse_args()
    args.data_dir = os.path.join(args.data_dir, args.dataset_name)

    main(args)