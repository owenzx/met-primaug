import nltk
from pprint import pprint
import pickle as pkl
import os

ROOT_FOLDER = os.path.dirname(os.path.realpath(__file__))

data = {}

tag2token = {}
token2tag = {}

max_len_x, max_len_y = 0, 0
for split in ("train", "dev", "test", "gen"):
    split_data = []
    for l in open(f"{ROOT_FOLDER}/COGS/cogs/{split}.tsv", "r").readlines():
        text, sparse, _ = l.split("\t")
        text, sparse = (text.split(" "), sparse.split(" "))
        max_len_x = max(len(text), max_len_x)
        max_len_y = max(len(sparse), max_len_y)

        token_tags = nltk.pos_tag(text)

        for tok, tag in token_tags:
            if tok not in token2tag:
                token2tag[tok] = {}
            if tag not in token2tag[tok]:
                token2tag[tok][tag] = 1
            else:
                token2tag[tok][tag] += 1


        for tok, tag in token_tags:
            if tag in tag2token:
                tag2token[tag].add(tok)
            else:
                tag2token[tag] = set()
                tag2token[tag].add(tok)



pprint(tag2token)



def get_tag_w_max_num(d):
    tag_count_list = list(d.items())
    max_tag = max(tag_count_list,key=lambda x:x[1])[0]
    return max_tag

for tok, tok_dict in token2tag.items():
    token2tag[tok] = get_tag_w_max_num(tok_dict)


rev_token2tag = {}
for tok, tag in token2tag.items():
    if tag not in rev_token2tag:
        rev_token2tag[tag] = set()
    rev_token2tag[tag].add(tok)

pprint(rev_token2tag)

with open(f'{ROOT_FOLDER}/tags/cogs_word2tag.pkl', 'wb') as fw:
    pkl.dump(token2tag, fw)

with open(f'{ROOT_FOLDER}/tags/cogs_tag2word.pkl', 'wb') as fw:
    pkl.dump(rev_token2tag, fw)
