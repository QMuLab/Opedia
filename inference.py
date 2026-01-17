
# %%
import copy
import pickle
import gc
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from typing import List, Tuple, Dict, Union, Optional
import warnings
import pandas as pd
# from . import asyn
import pickle
import torch
from anndata import AnnData
import scanpy as sc
# import scvi
import seaborn as sns
import numpy as np
from scipy.sparse import issparse
import matplotlib.pyplot as plt
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from torchtext.vocab import Vocab
from torchtext._torchtext import (
    Vocab as VocabPybind,
)
from sklearn.metrics import confusion_matrix

sys.path.insert(0, "../")
import scgpt as scg
from scgpt.model import TransformerModel, AdversarialDiscriminator
from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
from scgpt.loss import (
    masked_mse_loss,
    masked_relative_error,
    criterion_neg_log_bernoulli,
)
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.preprocess import Preprocessor
from scgpt import SubsetsBatchSampler
from scgpt.utils import set_seed, category_str2int, eval_scib_metrics

sc.set_figure_params(figsize=(6, 6))
os.environ["KMP_WARNINGS"] = "off"
warnings.filterwarnings('ignore')


# settings for input and preprocessing
pad_token = "<pad>"
special_tokens = [pad_token, "<cls>", "<eoc>"]
mask_ratio = 0.0
mask_value = "auto"  # for masked values, now it should always be auto

include_zero_gene = False  # if True, include zero genes among hvgs in the training
max_seq_len = 1201
n_bins = 51

# input/output representation flag
input_style = "binned"  # "normed_raw", "log1p", or "binned"
output_style = "binned"  # "normed_raw", "log1p", or "binned"

# settings for training
MLM = False  # whether to use masked language modeling, currently it is always on.
CLS = True  # celltype classification objective
ADV = False  # Adversarial training for batch correction
CCE = False  # Contrastive cell embedding objective
MVC = False  # Masked value prediction for cell embedding
ECS = 0.0 > 0  # Elastic cell similarity objective
DAB = False  # Domain adaptation by reverse backpropagation, set to 2 for separate optimizer
INPUT_BATCH_LABELS = False  # TODO: have these help MLM and MVC, while not to classifier
input_emb_style = "continuous"  # "category" or "continuous" or "scaling" # flag
cell_emb_style = "cls"  # "avg-pool" or "w-pool" or "cls"
adv_E_delay_epochs = 0  # delay adversarial training on encoder for a few epochs
adv_D_delay_epochs = 0
mvc_decoder_style = "inner product"
ecs_threshold = 0.0
dab_weight = 0.0

explicit_zero_prob = MLM and include_zero_gene  # whether explicit bernoulli for zeros
do_sample_in_train = False and explicit_zero_prob  # sample the bernoulli in training

per_seq_batch_sample = False

# settings for optimizer
lr = 1e-4  # TODO: test learning rate ratio between two tasks
lr_ADV = 1e-3  # learning rate for discriminator, used when ADV is True
batch_size = 32
eval_batch_size = 32
epochs = 3
schedule_interval = 1

# settings for the model
fast_transformer = False
fast_transformer_backend = "flash"  # "linear" or "flash"
embsize = 512  # embedding dimension
d_hid = 512  # dimension of the feedforward network in TransformerEncoder
nlayers = 12  # number of TransformerEncoderLayer in TransformerEncoder
nhead = 8  # number of heads in nn.MultiheadAttention
dropout = 0.2  # dropout probability

# logging
log_interval = 100  # iterations
save_eval_interval = 5  # epochs
do_eval_scib_metrics = True

# %% validate settings
assert input_style in ["normed_raw", "log1p", "binned"]
assert output_style in ["normed_raw", "log1p", "binned"]
assert input_emb_style in ["category", "continuous", "scaling"]

if input_style == "binned":
    if input_emb_style == "scaling":
        raise ValueError("input_emb_style `scaling` is not supported for binned input.")
elif input_style == "log1p" or input_style == "normed_raw":
    if input_emb_style == "category":
        raise ValueError(
            "input_emb_style `category` is not supported for log1p or normed_raw input."
        )

if input_emb_style == "category":
    mask_value = n_bins + 1
    pad_value = n_bins  # for padding gene expr values
    n_input_bins = n_bins + 2
else:
    mask_value = -1
    pad_value = -2
    n_input_bins = n_bins

if ADV and DAB:
    raise ValueError("ADV and DAB cannot be both True.")
DAB_separate_optim = True if DAB > 1 else False


vocab_file = '.\ckpt\dev_data_7ds-Mar07-14-37\vocab.json'
vocab = GeneVocab.from_file(vocab_file)
vocab.set_default_index(vocab["<pad>"])

state_dict = torch.load(".\ckpt\dev_data_7ds-Mar07-14-37\model.pt")

with open(".\ckpt\dev_data_7ds-Mar07-14-37\id2type.json", "r") as f:
    id2type = json.load(f)

id2type = {int(k): v for k, v in id2type.items()}




########################################################################################################################
# custom 

save_dir = Path(".\save")
save_dir.mkdir(parents=True, exist_ok=True)

adata = sc.read_h5ad('custom_file_path')  # 读取第一个h5ad
adata.var["gene_name"] = adata.var.index.tolist()

########################################################################################################################




adata.var["id_in_vocab"] = [
        1 if gene in vocab else -1 for gene in adata.var["gene_name"]
    ]
gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
print(
        f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
        f"in vocabulary of size {len(vocab)}."
    )
adata = adata[:, adata.var["id_in_vocab"] >= 0]

data_is_raw = False
freeze = False

preprocessor = Preprocessor(
    use_key="X",  # the key in adata.layers to use as raw data 
    filter_gene_by_counts=False,  # step 1
    filter_cell_by_counts=False,  # step 2
    normalize_total=None,  # 3. whether to normalize the raw data and to what sum
    result_normed_key="X_normed",  # the key in adata.layers to store the normalized data
    log1p=data_is_raw,  # 4. whether to log1p the normalized data
    result_log1p_key="X_log1p",
    subset_hvg=False,  # 5. whether to subset the raw data to highly variable genes
    hvg_flavor="seurat_v3" if data_is_raw else "cell_ranger",
    binning=n_bins,  # 6. whether to bin the raw data and to what number of bins
    result_binned_key="X_binned",  # the key in adata.layers to store the binned data
)

preprocessor(adata, batch_key=None)


input_layer_key = {  # the values of this map coorespond to the keys in preprocessing
    "normed_raw": "X_normed",
    "log1p": "X_normed",
    "binned": "X_binned",
}[input_style]
# all_counts = (
#     adata.layers[input_layer_key].A
#     if issparse(adata.layers[input_layer_key])
#     else adata.layers[input_layer_key]
# )
genes = adata.var["gene_name"].tolist()
gene_ids = np.array(vocab(genes), dtype=int)


class SeqDataset(Dataset):
    def __init__(self, data: Dict[str, torch.Tensor]):
        self.data = data

    def __len__(self):
        return self.data["gene_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}
    
# data_loader
def prepare_dataloader(
    data_pt: Dict[str, torch.Tensor],
    batch_size: int,
    shuffle: bool = False,
    intra_domain_shuffle: bool = False,
    drop_last: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    if num_workers == 0:
        num_workers = min(len(os.sched_getaffinity(0)), batch_size // 2)

    dataset = SeqDataset(data_pt)

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=True,
    )
    return data_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ntokens = len(vocab)  # size of vocabulary
model = TransformerModel(
    ntokens,
    embsize,
    nhead,
    d_hid,
    nlayers,
    nlayers_cls=3,
    n_cls=21 if CLS else 1,
    vocab=vocab,
    dropout=dropout,
    pad_token=pad_token,
    pad_value=pad_value,
    do_mvc=MVC,
    do_dab=DAB,
    use_batch_labels=INPUT_BATCH_LABELS,
    num_batch_labels=1,
    domain_spec_batchnorm=False,
    input_emb_style=input_emb_style,
    n_input_bins=n_input_bins,
    cell_emb_style=cell_emb_style,
    mvc_decoder_style=mvc_decoder_style,
    ecs_threshold=ecs_threshold,
    explicit_zero_prob=explicit_zero_prob,
    use_fast_transformer=fast_transformer,
    fast_transformer_backend=fast_transformer_backend,
    pre_norm=False,
)

model.to(device)


criterion_cls = nn.CrossEntropyLoss()

def evaluate(model: nn.Module, loader: DataLoader, return_raw: bool = False) -> float:
    """
    Evaluate the model on the evaluation data.
    """


    model.eval()
    total_loss = 0.0
    total_error = 0.0
    total_dab = 0.0
    total_num = 0
    predictions = []
    predictions_value = []
    collected_data = [] 

    all_cls_outputs = []
    all_cls_labels = [] 
    with torch.no_grad():
        for batch_data in loader:
            input_gene_ids = batch_data["gene_ids"].to(device)
            input_values = batch_data["values"].to(device)
            target_values = batch_data["target_values"].to(device)
            # batch_labels = batch_data["batch_labels"].to(device)
            # celltype_labels = batch_data["celltype_labels"].to(device)

            src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
            with torch.cuda.amp.autocast(enabled=True):
                output_dict = model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    # batch_labels=batch_labels if INPUT_BATCH_LABELS or config.DSBN else None,
                    CLS=CLS,  # evaluation does not need CLS or CCE
                    CCE=False,
                    MVC=False,
                    ECS=False,
                    do_sample=do_sample_in_train,
                    #generative_training = False,
                )
    
      
                # all_cls_labels.append(celltype_labels) 
                all_cls_outputs.append(output_dict["cls_output"])

                output_values = output_dict["cls_output"]
            
                # loss = criterion_cls(output_values, celltype_labels)



            softmax_values = F.softmax(output_values, dim=1)

            max_values, preds = softmax_values.max(1) 

            max_values = max_values.cpu().numpy() 
            preds = preds.cpu().numpy() 


         
            predictions.append(preds)
            predictions_value.append(max_values)
 



    # df = pd.DataFrame(collected_data, columns=["celltype_labels", "output_values"])
    # df.to_csv(save_to_file, index=False)
        # all_cls_labels = torch.cat(all_cls_labels, dim=0) 
        all_cls_outputs = torch.cat(all_cls_outputs, dim=0)
        # all_cls_labels = all_cls_labels.unsqueeze(1)
        # combined_output = torch.cat((all_cls_labels, all_cls_outputs), dim=1) 
        # flag
        # torch.save(combined_output, save_dir /"train_cls_outputs.pt")  # 保存到文件


    if return_raw:
        return np.concatenate(predictions, axis=0),np.concatenate(predictions_value, axis=0),all_cls_outputs

    return 



def test(model: nn.Module, adata: DataLoader) -> float:
    all_counts = (
        adata.layers[input_layer_key].A
        if issparse(adata.layers[input_layer_key])
        else adata.layers[input_layer_key]
    )

    # celltypes_labels = adata.obs["celltype_id"].tolist()  # make sure count from 0
    # celltypes_labels = np.array(celltypes_labels)
    # print 测试集的lable

    # batch_ids = adata.obs["batch_id"].tolist()
    # batch_ids = np.array(batch_ids)

    tokenized_test = tokenize_and_pad_batch(
        all_counts,
        gene_ids,
        max_len=max_seq_len,
        vocab=vocab,
        pad_token=pad_token,
        pad_value=pad_value,
        append_cls=True,  # append <cls> token at the beginning
        include_zero_gene=include_zero_gene,
    )

    input_values_test = random_mask_value(
        tokenized_test["values"],
        mask_ratio=mask_ratio,
        mask_value=mask_value,
        pad_value=pad_value,
    )

    test_data_pt = {
        "gene_ids": tokenized_test["genes"],
        "values": input_values_test,
        "target_values": tokenized_test["values"],
        # "batch_labels": torch.from_numpy(batch_ids).long(),
        # "celltype_labels": torch.from_numpy(celltypes_labels).long(),
    }

    test_loader = DataLoader(
        dataset=SeqDataset(test_data_pt),
        batch_size=eval_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=min(len(os.sched_getaffinity(0)), eval_batch_size // 2),
        pin_memory=True,
    )

    model.eval()
    predictions,predictions_value,all_cls_outputs = evaluate(
        model,
        loader=test_loader,
        return_raw=True,
    )

    # compute accuracy, precision, recall, f1
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    return predictions, predictions_value,all_cls_outputs





model.load_state_dict(state_dict)


best_model = model.eval()


predictions,predictions_value,all_cls_outputs = test(best_model, adata)

sorted_indices = torch.argsort(all_cls_outputs, dim=1, descending=True)  # [cell_num, num_cls]


id2type = {int(k): v for k, v in id2type.items()}
sorted_type_names = [[id2type[int(i)] for i in row] for row in sorted_indices.tolist()]

df = pd.DataFrame(sorted_type_names)
save_path = save_dir / "rst.csv"
df.to_csv(save_path, index=True, header=[f"rank_{i+1}" for i in range(df.shape[1])])