import os
import random
import re
import json
import torch
from torch import nn, optim
import torch.utils.data as torch_data
import numpy as np

from tqdm import tqdm

import seaborn as sns
import matplotlib.pyplot as plt

from absl import app, flags

from mutex import Vocab, RecordLoss
from meta_wrapper import MetaWrapper
from data import collate, eval_format, eval_format_probs, preds_to_refs_batch, encode_io_with_idx
from src import NoamLR
import hlog

import wandb

sns.set()
FLAGS = flags.FLAGS
flags.DEFINE_integer("dim", 512, "trasnformer dimension")
flags.DEFINE_integer("n_layers", 2, "number of rnn layers")
flags.DEFINE_integer("n_decoder_layers", 0, "number of rnn decoder layers")
flags.DEFINE_integer("n_batch", 512, "batch size")
flags.DEFINE_float("gclip", 0.5, "gradient clip")
flags.DEFINE_integer("n_epochs", 100, "number of training epochs")
flags.DEFINE_integer("beam_size", 5, "beam search size")
flags.DEFINE_integer("n_best", 30, "n_best_size for channel models")
flags.DEFINE_float("lr", 1.0, "learning rate")
flags.DEFINE_float("temp", 1.0, "temperature for samplings")
flags.DEFINE_float("dropout", 0.4, "dropout")
flags.DEFINE_string("load_model", "", "load pretrained model")
flags.DEFINE_integer("seed", 0, "random seed")
flags.DEFINE_bool("debug", False, "debug mode")
flags.DEFINE_bool("full_data", True, "full figure 2 experiments or simple col")
flags.DEFINE_bool("COGS", False, "COGS experiments")
flags.DEFINE_bool("TOY", False, "synthetic toy experiments")
flags.DEFINE_bool("regularize", False, "regularization")
flags.DEFINE_bool("SCAN", False, "SCAN experiments")
flags.DEFINE_bool("TRANSLATE", False, "TRANSLATE experiments")
flags.DEFINE_bool("bidirectional", False, "bidirectional encoders")
flags.DEFINE_bool("attention", True, "Source Attention")
flags.DEFINE_integer("warmup_steps", 4000, "noam warmup_steps")
flags.DEFINE_integer("valid_steps", 500, "validation steps")
flags.DEFINE_integer("max_step", 8000, "maximum number of steps")
flags.DEFINE_integer("tolarance", 5, "early stopping tolarance")
flags.DEFINE_integer("accum_count", 4, "grad accumulation count")
flags.DEFINE_bool("shuffle", True, "shuffle training set")
flags.DEFINE_bool("lr_schedule", True, "noam lr scheduler")
flags.DEFINE_string("scan_split", "", "around_right or jump or mcd1 or mcd2 or mcd3")
flags.DEFINE_bool("qxy", False, "train x|y model for reranking")
flags.DEFINE_bool("highdrop", False, "high drop mechanism")
flags.DEFINE_bool("highdroptest", False, "high drop at test")
flags.DEFINE_float("highdropvalue", 0.5, "high drop value")
flags.DEFINE_string("aligner", "", "alignment file by fastalign")
flags.DEFINE_bool("soft_align", False, "lexicon projection matrix")
flags.DEFINE_float("soft_temp", 0.2, "2 * temperature for soft lexicon")
flags.DEFINE_string("attention_type", "simple", "attention type used in the model")
flags.DEFINE_string("model_type", "lstm", "use lstm or transformer")
flags.DEFINE_string("transformer_config", "6layer", "transformer config")
flags.DEFINE_string("proj_name", "metacomp", "project name for wandb")
flags.DEFINE_string("exp_name", "metacomp", "exp name prefix for wandb")
flags.DEFINE_bool("train", False, "training mode")
flags.DEFINE_bool("cogs_perturbation", False, "controlled perturbation for cogs")
flags.DEFINE_bool("scan_perturbation", False, "controlled perturbation for scan")
flags.DEFINE_string("dataset_special_id", "", "identifier for speical datasets")

# additional arguments for meta training
flags.DEFINE_bool("meta", False, "use meta training")
flags.DEFINE_bool("unlikelihood", False, "use unlikelihood training")
flags.DEFINE_integer("n_inner_iter", 5, "number of update steps for the inner optimizer")
flags.DEFINE_float("mlm_prob", 0.15, "probability for masking")
flags.DEFINE_float("meta_loss_weight", 1.0, "mtl weight for the meta loss")
flags.DEFINE_float("ul_loss_weight", 1.0, "ul weight for the meta loss")
flags.DEFINE_string("meta_loss_type", None, "meta loss type")
flags.DEFINE_float("inner_lr", None, "learning rate for the inner opt, set to None to use the same as the outer lr")
flags.DEFINE_integer("multi_permutation", 1, "number of permutations in mx dataset")
flags.DEFINE_integer("multi_permutation_split", 1, "number of permutations in mx dataset, all in split files")
flags.DEFINE_string("special_train_data", None, "set to the name of special train split")
plt.rcParams['figure.dpi'] = 300

ROOT_FOLDER = os.path.dirname(os.path.realpath(__file__))
DEVICE = torch.device(("cuda" if torch.cuda.is_available() else "cpu"))




def prepare_data(FLAGS):

    vocab_x = Vocab()
    vocab_y = Vocab()
    references = None

    if FLAGS.SCAN:
        data = {}
        max_len_x, max_len_y = 0, 0
        reg = re.compile('^IN\:\s(.*?)\sOUT\: (.*?)$')
        if FLAGS.scan_split == "around_right":
            scan_file = "comp-data/SCAN/template_split/tasks_{}_template_around_right.txt"
        elif FLAGS.scan_split == "jump":
            scan_file = "comp-data/SCAN/add_prim_split/tasks_{}_addprim_jump.txt"
        elif FLAGS.scan_split == "dax":
            scan_file = "comp-data/SCAN/add_prim_split/tasks_{}_addprim_dax.txt"
        elif FLAGS.scan_split == "mcd1":
            scan_file = "comp-data/SCAN/mcd_split/tasks_{}_mcd1.txt"
        elif FLAGS.scan_split == "mcd2":
            scan_file = "comp-data/SCAN/mcd_split/tasks_{}_mcd2.txt"
        elif FLAGS.scan_split == "mcd3":
            scan_file = "comp-data/SCAN/mcd_split/tasks_{}_mcd3.txt"
        else:
            raise ValueError
        test_split_name = 'newtest'
        dev_split_name = 'gooddev'
        train_split_name = 'train'
        if FLAGS.special_train_data is not None:
            train_split_name = FLAGS.special_train_data

        splits = [train_split_name, dev_split_name, test_split_name]

        for split in splits:
            split_data = []
            line_idx = 0
            for l in open(f"{ROOT_FOLDER}/" + scan_file.format(split), "r").readlines():
                m = reg.match(l)
                inp, out = m.groups(1)
                inp, out = (inp.split(" "), out.split(" "))
                max_len_x = max(len(inp), max_len_x)
                max_len_y = max(len(out), max_len_y)
                for t in inp:
                    vocab_x.add(t)
                for t in out:
                    vocab_y.add(t)
                split_data.append(encode_io_with_idx((inp, out), vocab_x, vocab_y, line_idx))
                line_idx += 1
            data[split] = split_data

        train_items = data[train_split_name]
        val_items = data[dev_split_name]
        test_items = data[test_split_name]

        max_len_x += 1
        max_len_y += 1
        hlog.value("vocab_x len: ", len(vocab_x))
        hlog.value("vocab_y len: ", len(vocab_y))
        hlog.value("split lengths: ", [(k, len(v)) for (k, v) in data.items()])
    elif FLAGS.COGS:
        data = {}
        max_len_x, max_len_y = 0, 0

        old_gentest_split_name = 'gen'
        old_dev_split_name = 'dev'
        test_split_name = 'new_test'
        dev_split_name = 'dev_gen'
        train_split_name = 'train'
        if FLAGS.special_train_data is not None:
            train_split_name = FLAGS.special_train_data
        splits = [train_split_name, dev_split_name, test_split_name, old_dev_split_name, old_gentest_split_name]

        for split in splits:
            split_data = []
            line_idx = 0
            for l in open(f"{ROOT_FOLDER}/comp-data/COGS/cogs/{split}.tsv", "r").readlines():
                text, sparse, _ = l.split("\t")
                text, sparse = (text.split(" "), sparse.split(" "))
                max_len_x = max(len(text), max_len_x)
                max_len_y = max(len(sparse), max_len_y)
                for t in text:
                    vocab_x.add(t)
                for t in sparse:
                    vocab_y.add(t)
                split_data.append(encode_io_with_idx((text, sparse), vocab_x, vocab_y, line_idx))
                line_idx += 1
            data[split] = split_data


        max_len_x += 1
        max_len_y += 1
        hlog.value("vocab_x len: ", len(vocab_x))
        hlog.value("vocab_y len: ", len(vocab_y))
        hlog.value("split lengts: ", [(k, len(v)) for (k, v) in data.items()])
        train_items = data[train_split_name]
        val_items = data[dev_split_name]
        test_items = data[test_split_name]
    elif FLAGS.TOY:
        data = {}
        max_len_x, max_len_y = 0, 0
        for split in ["train", "dev", "gen"]:
            split_data = []
            with open(f"{ROOT_FOLDER}/comp-data/TOY/{split}.jsonl") as fr:
                line_idx = 0
                for line in fr.readlines():
                    example = json.loads(line)
                    src = example['src'].split(' ')
                    tgt = example['tgt'].split(' ')
                    max_len_x = max(max_len_x, len(src))
                    max_len_y = max(max_len_y, len(tgt))
                    for w in src:
                        vocab_x.add(w)
                    for w in tgt:
                        vocab_y.add(w)
                    split_data.append(encode_io_with_idx((src, tgt), vocab_x, vocab_y, line_idx))
                    line_idx += 1
            data[split] = split_data
        train_items = data['train']
        val_items = data['dev']
        test_items = data['gen']

    else:
        raise ValueError

    if FLAGS.meta:
        vocab_x.add_mask()
        vocab_y.add_mask()

    return vocab_x, vocab_y, max_len_x, max_len_y, train_items, val_items, test_items, references, data


def train(opt, model, train_dataset, val_dataset, references=None, full_exp_name=None, double_batch_training=False, additional_data=None):

    if FLAGS.lr_schedule:
        scheduler = NoamLR(opt, FLAGS.dim, warmup_steps=FLAGS.warmup_steps)
    else:
        scheduler = None


    train_loader = torch_data.DataLoader(
        train_dataset,
        batch_size=FLAGS.n_batch,
        shuffle=FLAGS.shuffle,
        collate_fn=collate
    )

    if double_batch_training:
        aux_train_loader = torch_data.DataLoader(
            train_dataset,
            batch_size=FLAGS.n_batch,
            shuffle=FLAGS.shuffle,
            collate_fn=collate
        )
        aux_data_iter = iter(aux_train_loader)
    else:
        aux_batch = None




    def get_next_batch(data_loader, data_iter):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(data_loader)
            batch = next(data_iter)
        return batch, data_iter


    tolarance = FLAGS.tolarance
    best_f1 = best_acc = -np.inf
    best_loss = np.inf
    steps = accum_steps = 0
    got_nan = False
    is_running = lambda: not got_nan and accum_steps < FLAGS.max_step and tolarance > 0
    while is_running():
        train_loss = train_batches = 0
        train_only_loss = train_other_loss = 0
        opt.zero_grad()
        recorder = RecordLoss()
        for train_sample in tqdm(train_loader):
            inp, out, lens, index = train_sample
            if not is_running():
                break
            if FLAGS.meta or FLAGS.unlikelihood:
                if double_batch_training:
                    aux_batch, aux_data_iter = get_next_batch(aux_train_loader, aux_data_iter)
                    aux_batch = [x.to(DEVICE) if x is not None else None for x in aux_batch]
                all_train_losses = model.train_forward(inp.to(DEVICE), out.to(DEVICE), lens=lens.to(DEVICE), recorder=recorder, accum_count = FLAGS.accum_count, current_lr = scheduler.get_lr(), aux_batch=aux_batch)
                train_batch_loss, batch_train_only_loss, batch_other_loss = all_train_losses
            else:
                train_batch_loss = nll = model(inp.to(DEVICE), out.to(DEVICE), lens=lens.to(DEVICE), recorder=recorder)
                batch_train_only_loss = train_batch_loss
                batch_other_loss= 0
            steps += 1
            loss = train_batch_loss / FLAGS.accum_count
            accum_train_only_loss = batch_train_only_loss / FLAGS.accum_count
            accum_other_loss = batch_other_loss / FLAGS.accum_count
            if not (FLAGS.meta or FLAGS.unlikelihood):
                loss.backward()
            train_loss += (loss.detach().item() * FLAGS.accum_count)
            if type(accum_train_only_loss) is not float:
                train_only_loss += (accum_train_only_loss.detach().item() * FLAGS.accum_count)
            else:
                train_only_loss += (accum_train_only_loss * FLAGS.accum_count)
            if type(accum_other_loss) is not float:
                train_other_loss += (accum_other_loss.detach().item() * FLAGS.accum_count)
            else:
                train_other_loss += (accum_other_loss * FLAGS.accum_count)
            train_batches += 1
            if steps % FLAGS.accum_count == 0:
                accum_steps += 1
                gnorm = nn.utils.clip_grad_norm_(model.parameters(), FLAGS.gclip)
                if not np.isfinite(gnorm.cpu()):
                    got_nan = True
                    print("=====GOT NAN=====")
                    break
                opt.step()
                opt.zero_grad()

                if scheduler is not None:
                    scheduler.step()


                if accum_steps % FLAGS.valid_steps == 0:
                    with hlog.task(accum_steps):
                        wandb.log({"train" + "/loss": train_loss / train_batches})
                        hlog.value("curr loss", train_loss / train_batches)
                        wandb.log({"train" + "/train_only_loss": train_only_loss / train_batches})
                        hlog.value("train only loss", train_only_loss / train_batches)
                        wandb.log({"train" + "/other_loss": train_other_loss / train_batches})
                        hlog.value("other loss", train_other_loss / train_batches)
                        acc, f1, val_loss = validate(model, val_dataset, references=references, split_name='val'.format(accum_steps))
                        val_acc = acc
                        model.train()
                        hlog.value("acc", acc)
                        hlog.value("f1", f1)
                        hlog.value("val_loss", val_loss)
                        if val_acc > best_acc:
                            best_acc = val_acc
                            tolarance = FLAGS.tolarance
                            torch.save(model, f"{full_exp_name}.best.model")
                        else:
                            tolarance -= 1
                        best_loss = min(best_loss, val_loss)
                        best_f1 = max(best_f1, f1)
                        best_acc = max(best_acc, acc)
                        hlog.value("best_loss", best_loss)
                        hlog.value("best_acc", best_acc)
                        hlog.value("best_f1", best_f1)

    wandb.log({"final_val" + "/acc": acc,
               "final_val" + "/f1": f1})
    wandb.log({"best_val" + "/acc": best_acc,
               "best_val" + "/f1": best_f1,
               "best_val" + "/loss": best_loss})
    hlog.value("final_acc", acc)
    hlog.value("final_f1", f1)
    hlog.value("best_acc", best_acc)
    hlog.value("best_f1", best_f1)
    hlog.value("best_loss", best_loss)
    return acc, f1




def validate(model, val_dataset, vis=False, beam=False, references=None, split_name=None):
    model.eval()
    val_loader = torch_data.DataLoader(
        val_dataset,
        batch_size=FLAGS.n_batch,
        shuffle=False,
        collate_fn=collate
    )
    total = correct = loss = tp = fp = fn = 0
    cur_references = []
    candidates = []
    with torch.no_grad():
        for inp, out, lens, index in tqdm(val_loader):
            input = inp.to(DEVICE)
            lengths = lens.to(DEVICE)
            pred, _ = model.sample(input,
                                   lens=lengths,
                                   temp=1.0,
                                   max_len=model.MAXLEN_Y,
                                   greedy=True,
                                   beam_size=FLAGS.beam_size * beam,
                                   calc_score=False)

            loss += model.pyx(input, out.to(DEVICE), lens=lengths).item() * input.shape[1]
            for i, seq in enumerate(pred):
                ref = out[:, i].numpy().tolist()
                ref = eval_format(model.vocab_y, ref)
                pred_here = eval_format(model.vocab_y, pred[i])
                if references is None:
                    cur_references.append([ref])
                else:
                    inpref = " ".join(model.vocab_x.decode(inp[0:lens[i], i].numpy().tolist()))
                    cur_references.append(references[inpref])

                candidates.append(pred_here)
                correct_here = pred_here == ref
                correct += correct_here
                tp_here = len([p for p in pred_here if p in ref])
                tp += tp_here
                fp_here = len([p for p in pred_here if p not in ref])
                fp += fp_here
                fn_here = len([p for p in ref if p not in pred_here])
                fn += fn_here
                total += 1
                if vis:
                    with hlog.task(total):
                        hlog.value("label", correct_here)
                        hlog.value("tp", tp_here)
                        hlog.value("fp", fp_here)
                        hlog.value("fn", fn_here)
                        inp_lst = inp[:, i].detach().cpu().numpy().tolist()
                        hlog.value("input", eval_format(model.vocab_x, inp_lst))
                        hlog.value("gold", ref)
                        hlog.value("pred", pred_here)

    wandb.log({split_name + "/loss": loss / total,
               split_name + "/acc": correct / total})

    acc = correct / total
    loss = loss / total
    if tp+fp > 0:
        prec = tp / (tp + fp)
    else:
        prec = 0
    rec = tp / (tp + fn)
    if prec == 0 or rec == 0:
        f1 = 0
    else:
        f1 = 2 * prec * rec / (prec + rec)
    hlog.value("acc", acc)
    hlog.value("f1", f1)
    wandb.log({split_name + "/f1": f1})
    return acc, f1, loss


def swap_io(items):
    return [(y, x) for (x, y) in items]


def main(argv):
    hlog.flags()
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)
    torch.cuda.manual_seed(FLAGS.seed)

    vocab_x, vocab_y, max_len_x, max_len_y, train_items, val_items, test_items, references, all_data = prepare_data(FLAGS)

    additional_data = None

    args_dict = FLAGS.flag_values_dict()
    full_exp_name = FLAGS.exp_name + '___' + 's{0}_b{1}_ly{2}_d{3}_lr{4}_dp{5}_b{6}_w{7}'.format(FLAGS.seed, FLAGS.n_batch * FLAGS.accum_count, FLAGS.n_layers, FLAGS.dim, FLAGS.lr, FLAGS.dropout, FLAGS.beam_size, FLAGS.meta_loss_weight)

    wandb.init(name=full_exp_name, project=FLAGS.proj_name, config=args_dict)

    if FLAGS.load_model == "":
        model = MetaWrapper(vocab_x,
                      vocab_y,
                      FLAGS.dim,
                      FLAGS.dim,
                      max_len_x=max_len_x,
                      max_len_y=max_len_y,
                      n_layers=FLAGS.n_layers,
                      self_att=False,
                      attention=FLAGS.attention,
                      dropout=FLAGS.dropout,
                      temp=FLAGS.temp,
                      bidirectional=FLAGS.bidirectional,
                      model_type = FLAGS.model_type,
                      transformer_config = FLAGS.transformer_config,
                      attention_type = FLAGS.attention_type,
                      n_decoder_layers = FLAGS.n_decoder_layers,
                      meta = FLAGS.meta,
                      unlikelihood = FLAGS.unlikelihood,
                      n_inner_iter = FLAGS.n_inner_iter,
                      mlm_prob= FLAGS.mlm_prob,
                      meta_loss_weight = FLAGS.meta_loss_weight,
                      ul_loss_weight = FLAGS.ul_loss_weight,
                      meta_loss_type = FLAGS.meta_loss_type,
                      cogs_perturbation = FLAGS.cogs_perturbation,
                      scan_perturbation = FLAGS.scan_perturbation,
                      inner_lr = FLAGS.inner_lr).to(DEVICE)
        double_batch_training = model.use_double_batch
    else:
        model = torch.load(FLAGS.load_model)

    if FLAGS.model_type == 'transformer':
        FLAGS.dim = model.pyx.output_dim

    if FLAGS.train:
        with hlog.task("train model"):
            opt = optim.Adam(model.pyx.parameters(), lr=FLAGS.lr, betas=(0.9, 0.998))

            acc, f1 = train(opt, model, train_items, val_items, references=references, full_exp_name=full_exp_name,
                                    double_batch_training=double_batch_training, additional_data=additional_data)
        torch.save(model, f"{full_exp_name}.final.model")



    if not FLAGS.train:
        with hlog.task("val evaluation"):
            validate(model, val_items, vis=True, references=references, split_name="val")

    with hlog.task("train evaluation"):
        validate(model, train_items, vis=False, references=references, split_name="train_eval")

    with hlog.task("test evaluation (greedy)"):
        validate(model, test_items, vis=True, beam=False, references=references, split_name="test(greedy)")


    if FLAGS.train:
        model = torch.load(f"{full_exp_name}.best.model")

        with hlog.task("train evaluation"):
            validate(model, train_items, vis=False, references=references, split_name="best_train_eval")

        with hlog.task("test evaluation (greedy)"):
            validate(model, test_items, vis=True, beam=False, references=references, split_name="best_test(greedy)")



if __name__ == "__main__":
    app.run(main)
