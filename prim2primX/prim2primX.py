import argparse
import os
import json
import collections
import logging
import sys
import csv
from tqdm import tqdm
import random
import numpy as np

sys.path.append('..')

def load_scan_file(data_file):
    fid = open(data_file, 'r')
    lines = fid.readlines()
    fid.close()
    lines = [l.strip() for l in lines]
    lines = [l.lstrip('IN: ') for l in lines]
    commands = [l.split(' OUT: ') for l in lines]
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

def write_cogs_data(data, data_path):
    with open(data_path, 'w') as tsv_file:
        writer = csv.writer(tsv_file, delimiter="\t")
        for row in data:
            writer.writerow([row['src'], row['trg'], row['type']])

def mutate_primitives_by_aligner(input_str, output_str, alignment, num_extra_actions, num_mutation, dataset_name,
                                 sampling=True):
    """
    Args:
        input_str:
        output_str:
        alignment:
        num_extra_actions:
        num_mutation:
        dataset_name:
        sampling: If True, randomly mutate a primitive to itself or primX; return "num_mutation" mutated examples.
                If False, enumerate all possible permutation of primitive mutation. For example, given "walk and look",
                for each primitive (e.g., walk), we enumerate all possible mutations: walk, walk0, walk1, etc;
                return all possible mutated examples

    Returns:

    """
    input_tok, output_tok = input_str.split(' '), output_str.split(' ')
    candidates = []
    # trg_to_src = {}
    augmented = []
    for (src_prim,mapped) in list(alignment.items()):
        trg_prim = list(mapped.keys())[0]
        if src_prim in input_tok:
            assert trg_prim in output_str, (trg_prim, output_str)
            candidates.append((src_prim, trg_prim))

    if len(candidates) == 0:
        return []

    all_mutated_srcs = []
    if sampling:
        for i in range(num_mutation*2):
            total_mutation = 0
            mutated_src, mutated_trg = input_tok, output_tok

            for src_prim, trg_prim in candidates:

                append_id = random.randint(0, num_extra_actions)
                if append_id == 0:  # Don't mutate this primitive
                    continue

                total_mutation += 1
                mutated_src = [x if x != src_prim else src_prim + str(append_id) for x in mutated_src]
                mutated_trg = [x if x != trg_prim else trg_prim + str(append_id) for x in mutated_trg]

            mutated_src, mutated_trg = ' '.join(mutated_src), ' '.join(mutated_trg)
            if mutated_src not in all_mutated_srcs:
                all_mutated_srcs.append(mutated_src)
                if total_mutation > 0:
                    if dataset_name == 'cogs':
                        augmented.append({'src': mutated_src, 'trg': mutated_trg, 'type': 'augmented_train'})
                    elif dataset_name == 'scan':
                        dp_line = 'IN: ' + mutated_src + ' OUT: ' + mutated_trg + '\n'
                        augmented.append(dp_line)
            if len(augmented) >= num_mutation:
                break
        return augmented[:num_mutation]
    else:
        assert dataset_name != "COGS", "Must do sampling with COGS, enumerate all possible permutation is too expensive"
        # if dataset_name == 'SCAN' and :
        #     assert len(candidates) <= 2

        num_permutation = np.power(num_extra_actions+1, len(candidates))
        x_permutations = np.zeros((num_permutation, len(candidates)))

        for i in range(num_permutation):
            for ic in range(len(candidates)):
                for ip in range(num_extra_actions):
                    if np.floor(i / np.power(num_extra_actions+1, ic)) % (num_extra_actions+1) == ip+1:
                        x_permutations[i][ic] = ip+1
                        break
                # elif np.floor(i / np.power(num_extra_actions+1, ic)) % (num_extra_actions+1) == 2:
                #     x_permutations[i][ic] = 2

        # print(x_permutations)
        for _ix, x_permutation in enumerate(x_permutations):
            mutated_src, mutated_trg = input_tok, output_tok
            total_mutation = 0
            for _ip, (src_prim, trg_prim) in enumerate(candidates):

                append_id = int(x_permutation[_ip])
                if append_id == 0:
                    continue
                total_mutation += 1

                mutated_src = [x if x != src_prim else src_prim + str(append_id) for x in mutated_src]
                mutated_trg = [x if x != trg_prim else trg_prim + str(append_id) for x in mutated_trg]

            mutated_src, mutated_trg = ' '.join(mutated_src), ' '.join(mutated_trg)
            if mutated_src not in all_mutated_srcs:
                all_mutated_srcs.append(mutated_src)
                if total_mutation > 0:
                    if dataset_name == 'cogs':
                        augmented.append({'src': mutated_src, 'trg': mutated_trg, 'type': 'augmented_train'})
                    elif dataset_name == 'scan':
                        dp_line = 'IN: ' + mutated_src + ' OUT: ' + mutated_trg + '\n'
                        augmented.append(dp_line)
        return augmented

def expand_data_by_substituting_primitives(all_data, alignment_dict, num_extra_actions, num_mutation, dataset_name,
                                           sampling=True):
    expanded_data = []

    for idx, dp in tqdm(enumerate(all_data), total=len(all_data)):
        if dataset_name == 'scan':
            src, trg = dp[0], dp[1]
        elif dataset_name == 'cogs':
            src, trg = dp['src'], dp['trg']
        else:
            raise NotImplementedError
        mutated_examples = mutate_primitives_by_aligner(
            src, trg, alignment_dict, num_extra_actions, num_mutation, dataset_name, sampling
        )
        if dataset_name == 'scan':
            expanded_data.append('IN: ' + src + ' OUT: ' + trg + '\n')
        expanded_data += mutated_examples

    return expanded_data

def expand_vocab(src_fname, trg_fname, num_extra_actions, alignment_dict):
    with open(src_fname, 'r') as f:
        src_vocab = f.read().splitlines()
    with open(trg_fname, 'r') as f:
        trg_vocab = f.read().splitlines()

    for (src_prim, mapped) in list(alignment_dict.items()):
        trg_prim = list(mapped.keys())[0]
        for i in range(num_extra_actions):
            src_vocab.append(src_prim+str(i+1))
            trg_prim_mut = trg_prim+str(i+1)
            if trg_prim_mut not in trg_vocab:
                trg_vocab.append(trg_prim_mut)

    return src_vocab, trg_vocab


def main():
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
        default="SCAN",
        choices=["SCAN", "COGS"]
    )
    parser.add_argument(
        "--num_extra_actions",
        type=int,
        default=2,
        help="The number of mutated primitives per original primitives. If equals 2, then for every prim we can have prim1, prim2"
    )
    parser.add_argument(
        "--num_mutation_per_example",
        type=int,
        default=None,
        help="number of mutated example per original example."
    )
    parser.add_argument(
        "--aligner_file",
        type=str
    )

    args = parser.parse_args()

    if args.dataset_name == 'COGS':
        args.data_dir = os.path.join(args.data_dir, 'cogs', args.dataset_name)
        data_file = os.path.join(args.data_dir, 'train.tsv')
        all_train_data = load_cogs_data(data_file)
        word_alignment_dict = json.load(open(os.path.join(args.data_dir, args.aligner_file)))

        augmented_data = expand_data_by_substituting_primitives(
            all_train_data,
            alignment_dict=word_alignment_dict,
            num_extra_actions=args.num_extra_actions,
            num_mutation=args.num_mutation_per_example,
            dataset_name='cogs'
        )
        cogs_write_path = os.path.join(args.data_dir, 'train' + '_primx'+str(args.num_extra_actions) + \
                                       's' + str(args.num_mutation_per_example) + '.tsv')
        write_cogs_data(all_train_data + augmented_data, cogs_write_path)
        print('Processed %d examples for train set.' % (len(all_train_data)+len(augmented_data)))

    elif args.dataset_name == 'SCAN':
        if args.split == "addprim_jump":
            args.data_dir = os.path.join(args.data_dir, args.dataset_name, 'add_prim_split')
            data_file = os.path.join(args.data_dir, 'tasks_train_addprim_jump.txt')
        elif args.split == "around_right":
            args.data_dir = os.path.join(args.data_dir, args.dataset_name, 'template_split')
            data_file = os.path.join(args.data_dir, 'tasks_train_template_around_right.txt')
        all_train_data = load_scan_file(data_file)
        word_alignment_dict = json.load(open(os.path.join(args.data_dir, args.aligner_file)))
        augmented_data = expand_data_by_substituting_primitives(
            all_train_data,
            alignment_dict=word_alignment_dict,
            num_extra_actions=args.num_extra_actions,
            num_mutation=args.num_mutation_per_example,
            dataset_name='scan',
            sampling=False if args.num_mutation_per_example is None else True
        )

        print('Processed %d examples for train set.' % (len(augmented_data)))
        with open(os.path.join(args.data_dir, f'tasks_train_primx{str(args.num_extra_actions)}_{args.split}.txt'), 'w') as f:
            f.writelines(augmented_data)

if __name__ == "__main__":
    main()