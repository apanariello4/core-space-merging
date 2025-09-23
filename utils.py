import argparse
import math
import os
import pdb
import random
import string
from collections import OrderedDict, defaultdict
from copy import deepcopy
from inspect import getmembers, isfunction
from pathlib import Path

import clip
import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from peft.utils import get_peft_model_state_dict
from sklearn.model_selection import train_test_split
from torch.cuda.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification
from models.huggingface_clip import get_model_from_config

CONCEPT_TASKS = list(string.ascii_uppercase)


##########################################################################################################################
######################################################### CLASSES ########################################################
##########################################################################################################################

def recursively_setattr(model, key, new_module):
    """Recursively set an attribute from the model. Supports layer sequences < 20 layers deep."""
    stages = key.split('.')
    x = getattr(model, stages[0])
    for stage in stages[1:-1]:
        if stage in [str(i) for i in range(20)]:
            x = x[int(stage)]
            continue
        x = getattr(x, stage)
    setattr(x, stages[-1], new_module)


def recursively_getattr(model, key):
    """Recursively get an attribute from the model. Supports layer sequences < 20 layers deep."""
    stages = key.split('.')
    x = model
    for stage in stages:
        if stage in [str(i) for i in range(20)]:
            x = x[int(stage)]
            continue
        x = getattr(x, stage)
    return x


class LoRAABLayer(nn.Module):
    """Combine LoRA AB parameters in a Huggigface ViT layer."""

    def __init__(self, linear):
        super().__init__()
        self.linear_weight = linear.weight
        self.linear_bias = linear.bias
        self.AB_weight = nn.Parameter(linear.lora_B.default.weight.data @ linear.lora_A.default.weight.data)

    def forward(self, x):
        linear_out = F.linear(x, self.linear_weight, self.linear_bias)
        lora_out = F.linear(x, self.AB_weight)
        return linear_out + lora_out


class LinearLayer(nn.Module):
    """Combine LoRA AB parameters in a Huggigface ViT layer and inject them into the original weights."""

    def __init__(self, linear):
        super().__init__()
        AB = linear.lora_B.default.weight.data @ linear.lora_A.default.weight.data
        self.linear_weight = nn.Parameter(linear.weight.data + AB)
        self.linear_bias = linear.bias

    def forward(self, x):
        linear_out = F.linear(x, self.linear_weight, self.linear_bias)
        return linear_out


def combine_lora_layers(model):
    """Combine LoRA AB layers in a Hugging Face ViT model."""
    for i in tqdm(range(len(model.vision_model.base_model.model.encoder.layers))):
        header = f'vision_model.base_model.model.encoder.layers.{i}'
        # Query module
        query_module = recursively_getattr(model, f'{header}.self_attn.q_proj')
        recursively_setattr(
            model, f'{header}.self_attn.q_proj',
            LinearLayer(query_module)
        )
        # Key module
        key_module = recursively_getattr(model, f'{header}.self_attn.k_proj')
        recursively_setattr(
            model, f'{header}.self_attn.k_proj',
            LinearLayer(key_module)
        )
        # Value module
        value_module = recursively_getattr(model, f'{header}.self_attn.v_proj')
        recursively_setattr(
            model, f'{header}.self_attn.v_proj',
            LinearLayer(value_module)
        )
        # Output module
        output_module = recursively_getattr(model, f'{header}.self_attn.out_proj')
        recursively_setattr(
            model, f'{header}.self_attn.out_proj',
            LinearLayer(output_module)
        )
    return model


class SpoofModel(torch.nn.Module):
    """wrap model, allow for multiple forward passes at once."""

    def __init__(self, models):
        super().__init__()
        self.models = models

    def forward(self, x):
        """Call all models returning list of their outputs."""
        return [model(x) for model in self.models]

    def parameters(self):
        """Return list of parameters from first model."""
        return self.models[0].parameters()


class EarlyStopper:
    # Copied from: https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch.
    def __init__(self, patience=1, min_delta=0, by_loss=False):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.max_validation_acc = -np.inf

    def early_stop(self, validation_acc):
        if validation_acc > self.max_validation_acc:
            self.max_validation_acc = validation_acc
            self.counter = 0
        elif validation_acc < (self.max_validation_acc - self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def assign_learning_rate(param_group, new_lr):
    """Assign a new learning rate to a parameter group."""
    param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    """Warmup learning rate schedule."""
    return base_lr * (step + 1) / warmup_length


def cosine_lr(optimizer, base_lrs, warmup_length, steps):
    """Cosine learning rate schedule."""
    if not isinstance(base_lrs, list):
        base_lrs = [base_lrs for _ in optimizer.param_groups]
    assert len(base_lrs) == len(optimizer.param_groups)

    def _lr_adjuster(step):
        for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
            if step < warmup_length:
                lr = _warmup_lr(base_lr, warmup_length, step)
            else:
                e = step - warmup_length
                es = steps - warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
            assign_learning_rate(param_group, lr)
    return _lr_adjuster


def step_lr(optimizer, base_lrs, start_lr, warmup_length, steps):
    """Step learning rate schedule."""
    if not isinstance(base_lrs, list):
        base_lrs = [base_lrs for _ in optimizer.param_groups]
    assert len(base_lrs) == len(optimizer.param_groups)

    def _lr_adjuster(step):
        for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
            if step < warmup_length:
                lr = start_lr
            else:
                e = step - warmup_length
                es = steps - warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
            assign_learning_rate(param_group, lr)
    return _lr_adjuster

##########################################################################################################################
################################################## TRAIN/EVAL FUNCTIONS ##################################################
##########################################################################################################################


def evaluate_logits(model, loader, device, mask_class=None, eval=True):
    """Evaluate a model trained with standard CE on a dataset."""
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    for batch in tqdm(loader, 'Evaluating model'):
        batch.to(device)
        with torch.no_grad():
            outputs = model(**batch)
            if mask_class is not None:
                if eval:
                    outputs.logits[:, mask_class] = -np.inf
                else:
                    outputs.logits[:, mask_class] = -1e10
            predictions = outputs.logits.argmax(dim=-1)
            total += batch["labels"].size(0)
            correct += (predictions == batch["labels"].to(device)).sum().item()
    return correct / total


# evaluates accuracy
def evaluate_cliphead(
        model, loader, class_vectors, remap_class_idxs=None,
        return_confusion=False, task_info=None, return_loss=False, silent=False):
    """Evaluate a model with a cliphead on a dataset."""
    model.eval()
    correct = 0
    total = 0

    totals = np.array([0] * class_vectors.shape[0])
    corrects = np.array([0] * class_vectors.shape[0])

    device = get_device(model)
    losses = []
    loss_fn = CrossEntropyLoss()
    with torch.no_grad(), autocast():
        for inputs, labels in tqdm(loader, 'Evaluating CLIP head model', disable=silent):
            encodings = model(inputs.to(device))
            normed_encodings = encodings / encodings.norm(dim=-1, keepdim=True)

            if task_info is not None:
                task_map = task_info['task_map']
                data_label_task = task_map[labels].to(device)
                task_features = torch.stack(task_info['task_features'], dim=0).transpose(-1, -2)[data_label_task]
                outputs = torch.einsum('ij,ijk->ik', normed_encodings, task_features)
                remap_class_idxs = task_info['remap_class_idxs']
            else:
                outputs = normed_encodings @ class_vectors.T
            pred = outputs.argmax(dim=1)
            if remap_class_idxs is not None:
                remapped_labels = remap_class_idxs[labels]
            else:
                remapped_labels = labels
            loss = loss_fn(outputs, remapped_labels.to(device))
            losses += [loss.item()]

            for gt, p in zip(labels, pred):
                if remap_class_idxs is not None:
                    idx = gt
                    gt = remap_class_idxs[gt]
                else:
                    idx = gt

                is_correct = (gt == p).item()
                correct += is_correct

                if return_confusion:
                    totals[idx] += 1
                    corrects[idx] += is_correct

            total += encodings.shape[0]

    overall_loss = np.mean(losses)

    if return_confusion:
        return correct / sum(totals), list(map(lambda a: a[0] / a[1], zip(corrects, totals)))
    else:
        if return_loss:
            return correct / total, overall_loss
        return correct / total

def get_clip_features(
        model, loader, silent=False):
    """Evaluate a model with a cliphead on a dataset."""
    model.eval()


    device = 'cuda'
    print(f"Extracting features on {device}")
    model.to(device)
    with torch.no_grad(), autocast():
        all_encodings = []
        all_labels = []
        for inputs, labels in tqdm(loader, 'Extracting features', disable=silent):
            encodings = model(inputs.to(device))
            normed_encodings = encodings / encodings.norm(dim=-1, keepdim=True)

            all_encodings.append(normed_encodings.cpu())
            all_labels.append(labels.cpu())

    return torch.cat(all_encodings, dim=0), torch.cat(all_labels, dim=0)


def evaluate_cliphead_joint(
        model, loader, class_vectors, aux_class_map=None):
    """Evaluate a model with a cliphead in the Joint setting."""
    model.eval()

    topk_counts = {i: 0 for i in [1, 3, 5, 10]}

    total = 0
    device = 'cuda'
    model_confusions = np.zeros((class_vectors.shape[0], class_vectors.shape[0]))

    with torch.no_grad():
        for inputs, labels in tqdm(loader, 'Evaluating CLIP head model'):
            encodings = model(inputs.to(device))
            if isinstance(encodings, list):
                normed_encodings = torch.stack(
                    [encoding / encoding.norm(dim=-1, keepdim=True) for encoding in encodings], dim=0
                )  # [N, B, D]
                outputs = (normed_encodings.to(class_vectors.device) @ class_vectors.T)  # [N, B, C]
                outputs = outputs.max(dim=0).values  # [B]
            else:
                normed_encodings = encodings / encodings.norm(dim=-1, keepdim=True)
                outputs = (normed_encodings @ class_vectors.T)  # [B, C]

            preds = outputs.argsort(dim=1, descending=True)
            # Map dataset class labels to new space
            if aux_class_map is not None:
                remapped_labels = aux_class_map[labels]
            else:
                remapped_labels = labels
            for gt, instance_preds in zip(remapped_labels, preds):
                gt_loc = torch.argwhere(instance_preds == gt).item()
                for k in topk_counts:
                    if gt_loc < k:
                        topk_counts[k] += 1
                model_confusions[gt, instance_preds[0]] += 1
            total += preds.shape[0]

    topk = {k: v / total for k, v in topk_counts.items()}

    return topk_counts, total, topk, model_confusions


def train_cliphead_lora(model, train_loader, test_loader, class_vectors, remap_class_idxs=None, eval_class_vectors=None, hyper_param_config=None):
    """Train a cliphead LoRA model."""
    epochs = hyper_param_config['epochs']
    optimizer = torch.optim.AdamW(model.parameters(), lr=hyper_param_config['lr'], weight_decay=hyper_param_config['wd'])
    ne_iters = len(train_loader)

    scheduler = cosine_lr(optimizer, hyper_param_config['lr'], hyper_param_config['warm_up'], epochs * ne_iters)

    scaler = GradScaler()
    loss_fn = CrossEntropyLoss(label_smoothing=hyper_param_config['label_smoothing'])

    device = get_device(model)

    losses = []
    acc = 0.
    pbar = tqdm(range(epochs), desc=f'finetuning, prev acc: {acc}: ')
    for epoch in pbar:
        model = model.train()
        for i, (inputs, labels) in tqdm(enumerate(train_loader), desc="iterating over epoch"):
            step = i + epoch * ne_iters
            optimizer.zero_grad(set_to_none=True)
            pdb.set_trace()
            # We assume input will be processed internally by model
            encodings = model(inputs.to(device))
            normed_encodings = encodings / encodings.norm(dim=-1, keepdim=True)
            logits = (100.0 * normed_encodings @ class_vectors.T)
            if remap_class_idxs is not None:
                remapped_labels = remap_class_idxs[labels].to(device)
            else:
                remapped_labels = labels.to(device)
            loss = loss_fn(logits, remapped_labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler(step)
            losses.append(loss.item())

        acc = evaluate_cliphead(model, test_loader, class_vectors=class_vectors, remap_class_idxs=remap_class_idxs)
        pbar.set_description(f'finetuning, prev acc: {acc}: ')
        print(f'Epoch {epoch}, Acc: {acc}')
    if eval_class_vectors is None:
        eval_class_vectors = class_vectors
    acc = evaluate_cliphead(model, test_loader, class_vectors=eval_class_vectors, remap_class_idxs=remap_class_idxs)
    return model, acc


##########################################################################################################################
############################################### EXPERIMENT CONFIG CREATION ###############################################
##########################################################################################################################

def prepare_data(config, device='cuda'):
    """ Load all dataloaders required for experiment. """
    if isinstance(config, list):
        return [prepare_data(c, device) for c in config]

    dataset_name = config['name']

    import dataset.configs as config_module
    data_config = deepcopy(getattr(config_module, dataset_name))
    data_config.update(config)
    data_config['device'] = device
    data_config['num_workers'] = min(os.cpu_count(), data_config['num_workers'])

    # NLI datasets
    if data_config['type'] == 'snli':
        from dataset.snli import prepare_test_loaders, prepare_train_loaders
        train_loaders = prepare_train_loaders(data_config)
        test_loaders = prepare_test_loaders(data_config)

    elif data_config['type'] == 'mnli':
        from dataset.mnli import prepare_test_loaders, prepare_train_loaders
        train_loaders = prepare_train_loaders(data_config)
        test_loaders = prepare_test_loaders(data_config)

    elif data_config['type'] == 'sick':
        from dataset.sick import prepare_test_loaders, prepare_train_loaders
        train_loaders = prepare_train_loaders(data_config)
        test_loaders = prepare_test_loaders(data_config)

    elif data_config['type'] == 'qnli':
        from dataset.qnli import prepare_test_loaders, prepare_train_loaders
        train_loaders = prepare_train_loaders(data_config)
        test_loaders = prepare_test_loaders(data_config)

    elif data_config['type'] == 'rte':
        from dataset.rte import prepare_test_loaders, prepare_train_loaders
        train_loaders = prepare_train_loaders(data_config)
        test_loaders = prepare_test_loaders(data_config)

    elif data_config['type'] == 'scitail':
        from dataset.scitail import prepare_test_loaders, prepare_train_loaders
        train_loaders = prepare_train_loaders(data_config)
        test_loaders = prepare_test_loaders(data_config)

    # -------------------------------------------------

    elif data_config['type'] == 'eurosat':
        from dataset.eurosat import prepare_test_loaders, prepare_train_loaders
        print('Loading Eurosat')
        train_loaders = prepare_train_loaders(data_config)
        test_loaders = prepare_test_loaders(data_config)
    elif data_config['type'] == 'stanford_cars':
        print('Loading Cars')
        from dataset.cars import prepare_test_loaders, prepare_train_loaders
        train_loaders = prepare_train_loaders(data_config)
        test_loaders = prepare_test_loaders(data_config)
    elif data_config['type'] == 'dtd':
        print('Loading DTD')
        from dataset.dtd import prepare_test_loaders, prepare_train_loaders
        train_loaders = prepare_train_loaders(data_config)
        test_loaders = prepare_test_loaders(data_config)
    elif data_config['type'] == 'mnist':
        print('Loading MNIST')
        from dataset.mnist import prepare_test_loaders, prepare_train_loaders
        train_loaders = prepare_train_loaders(data_config)
        test_loaders = prepare_test_loaders(data_config)
    elif data_config['type'] == 'gtsrb':
        print('Loading GTSRB')
        from dataset.gtsrb import prepare_test_loaders, prepare_train_loaders
        train_loaders = prepare_train_loaders(data_config)
        test_loaders = prepare_test_loaders(data_config)
    elif data_config['type'] == 'svhn':
        print('Loading SVHN')
        from dataset.svhn import prepare_test_loaders, prepare_train_loaders
        train_loaders = prepare_train_loaders(data_config)
        test_loaders = prepare_test_loaders(data_config)
    elif data_config['type'] == 'sun397':
        print('Loading SUN397')
        from dataset.sun397 import prepare_test_loaders, prepare_train_loaders
        train_loaders = prepare_train_loaders(data_config)
        test_loaders = prepare_test_loaders(data_config)
    elif data_config['type'] == 'resisc45':
        print('Loading RESISC45')
        from dataset.resisc45 import prepare_test_loaders, prepare_train_loaders
        train_loaders = prepare_train_loaders(data_config)
        test_loaders = prepare_test_loaders(data_config)
    else:
        raise NotImplementedError(config['type'])

    try:
        return {
            'train': train_loaders,
            'test': test_loaders
        }
    except BaseException:
        pdb.set_trace()


def replace_sd_keys(sd, original, new):
    new_sd = {}
    for key, val in sd.items():
        new_key = key.replace(original, new)
        new_sd[new_key] = val
    return new_sd


def prepare_param_handler(ft_config):
    """Load FT model parameter extractors"""
    if ft_config.get('type', None) == 'lora':
        from ft_handlers import LoRAHandler
        return LoRAHandler
    elif ft_config.get('type', None) == 'fft':
        from ft_handlers import FFTHandler
        return FFTHandler
    elif ft_config.get('type', None) == 'vera':
        from ft_handlers import VeRAHandler
        return VeRAHandler
    else:
        from ft_handlers import GeneralHandler
        return GeneralHandler


def check_sd_almost_equal(base, desired, okay_set=None):
    """Check if two state_dicts are almost equal."""
    for key in base.keys():
        if key not in desired:
            if okay_set is not None and key in okay_set:
                continue
            else:
                return False
    return True


def prepare_llama(config, device):
    """Load LLama models from config."""
    import psutil
    process = psutil.Process()

    bases = []
    peft_config = LoraConfig(task_type=config["peft_config"]["task_type"],
                             inference_mode=config["peft_config"]["inference_mode"],
                             r=config["peft_config"]["r"],
                             lora_alpha=config["peft_config"]["lora_alpha"],
                             lora_dropout=config["peft_config"]["lora_dropout"],
                             target_modules=config["peft_config"]["target_modules"],
                             )
    model_name_or_path = config['name']
    print("memory before loading ", process.memory_info().rss / 1024**2)
    ptm_model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path, return_dict=True, cache_dir=config['cachedir'], num_labels=3)
    base_model = get_peft_model(ptm_model, peft_config)
    for base_path in tqdm(config['bases'], desc="Preparing Models"):
        if base_path.endswith('.pt'):
            # Load just the LoRA state dict from local file
            lora_state_dict = torch.load(base_path, map_location='cpu')
            bases.append(lora_state_dict)
        else:
            # Load just the LoRA adapter state dict from HF
            adapter_model = PeftModel.from_pretrained(model=ptm_model, model_id=base_path)
            bases.append(get_peft_model_state_dict(adapter_model))

    if model_name_or_path == 'meta-llama/Meta-Llama-3-8B':
        ptm_model_path = 'hoffman-lab/KnOTS-Llama3_8B_lora_R16_pretrained_model'
        base_model = PeftModel.from_pretrained(model=ptm_model, model_id=ptm_model_path)
    else:
        base_model = get_peft_model(ptm_model, peft_config)  # Load LoRA model

    # base_model = get_peft_model(base_model, peft_config)  # Load LoRA model
    print("memory after loading ", process.memory_info().rss / 1024**2)
    return {
        'bases': bases,
        'new': base_model
    }


def prepare_hf_clip(config, device):
    """Load Hugging Face (HF) ViT models from config."""
    bases = []
    ranks = config['ft_config']['r']
    if isinstance(ranks, int):
        base_model = get_model_from_config(config, device)

    for i, base_path in enumerate(config['bases']):
        if isinstance(ranks, list):
            rank = ranks[i]
            config_i = deepcopy(config)
            config_i['ft_config']['r'] = rank
            if 'lora_alpha' in config['ft_config']:
                config_i['ft_config']['lora_alpha'] = config['ft_config']['lora_alpha'][i]
            base_model = get_model_from_config(config=config_i, device=device)
        if base_path.endswith('.pt'):
            # Load just the LoRA state dict from local file
            lora_state_dict = torch.load(base_path, map_location='cpu')
            # Add vision_model prefix to all keys
            lora_state_dict = replace_sd_keys(lora_state_dict, 'lora_model', 'vision_model')
            lora_state_dict = replace_sd_keys(lora_state_dict, 'linear_layer.', 'vision_head.')
            lora_state_dict = replace_sd_keys(lora_state_dict, '.base_layer', '')
            bases.append(lora_state_dict)
            print(f"Loaded model from {base_path}")
        else:
            # Load just the LoRA adapter state dict from HF
            lora_model = PeftModel.from_pretrained(model=base_model.vision_model.base_model.model, model_id=base_path)
            lora_state_dict = get_peft_model_state_dict(lora_model)
            # Add vision_model prefix to all keys
            lora_state_dict = replace_sd_keys(lora_state_dict, 'base_model', 'vision_model.base_model')
            bases.append(lora_state_dict)
            print(f"Loaded model from {base_path}")

    if isinstance(ranks, list):
        rank = ranks[i]
        config_i = deepcopy(config)
        config_i['ft_config']['r'] = rank
        if 'lora_alpha' in config['ft_config']:
            config_i['ft_config']['lora_alpha'] = config['ft_config']['lora_alpha'][i]
        base_model = get_model_from_config(config=config_i, device=device)
    else:
        base_model = get_model_from_config(config, device)

    return {
        'bases': bases,
        'new': base_model,
    }


class ModelWrapper(torch.nn.Module):
    def __init__(self, model, feature_dim, num_classes, normalize=False, initial_weights=None):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.classification_head = torch.nn.Linear(feature_dim, num_classes)
        self.normalize = normalize
        if initial_weights is None:
            initial_weights = torch.zeros_like(self.classification_head.weight)
            torch.nn.init.kaiming_uniform_(initial_weights, a=math.sqrt(5))
        self.classification_head.weight = torch.nn.Parameter(initial_weights.clone())
        self.classification_head.bias = torch.nn.Parameter(
            torch.zeros_like(self.classification_head.bias))

        # Note: modified. Get rid of the language part.
        if hasattr(self.model, 'transformer'):
            delattr(self.model, 'transformer')

    def forward(self, images, return_features=False):
        features = self.model.encode_image(images)
        if self.normalize:
            features = features / features.norm(dim=-1, keepdim=True)
        logits = self.classification_head(features)
        if return_features:
            return logits, features
        return logits


def get_model_from_sd(state_dict, base_model):
    feature_dim = state_dict['classification_head.weight'].shape[1]
    num_classes = state_dict['classification_head.weight'].shape[0]
    model = ModelWrapper(base_model, feature_dim, num_classes, normalize=True)
    for p in model.parameters():
        p.data = p.data.float()
    model.load_state_dict(state_dict)
    model = model.cuda()
    devices = [x for x in range(torch.cuda.device_count())]
    return torch.nn.DataParallel(model, device_ids=devices)


def prepare_oc_vit(config, device):
    bases = []
    base_model, preprocess = clip.load(config['oc_name'], device, jit=False)
    NUM_MODELS = config['num_models']
    model_paths = [os.path.join(config['dir'], f'model_{i}.pt') for i in range(NUM_MODELS)]
    for j, model_path in enumerate(model_paths):
        print(f"loading model #{j}")
        assert os.path.exists(model_path)
        state_dict = torch.load(model_path, map_location=torch.device(device))
        model = get_model_from_sd(state_dict, base_model)
        bases += [model]
    torch.load(config['pretrained_model_dir'], map_location=torch.device(device))
    new_model = get_model_from_sd(state_dict, base_model)
    print("All models have been loaded, we are ready to merge!")
    return {
        'bases': bases,
        'new': new_model
    }


def prepare_models(config, device='cuda'):
    """ Load all pretrained models in config. """
    if config['name'].startswith('oc_vit'):
        return prepare_oc_vit(config, device)
    elif config['name'].startswith('hf_clip'):
        return prepare_hf_clip(config, device)
    elif config['name'].startswith('meta-llama'):
        return prepare_llama(config, device)
    else:
        raise NotImplementedError(config['name'])


def get_merging_fn(name):
    """Get the merging function from name tag."""
    import merging_functions
    vector_fns = dict([(k.replace('_merging', ''), v) for (k, v) in getmembers(merging_functions, isfunction) if '_merging' in k])
    return vector_fns[name]


def get_mask_fn(name):
    """Get the masking function from name tag."""
    import masking_ops
    masking_fns = dict([(k.replace('_masking', ''), v) for (k, v) in getmembers(masking_ops, isfunction) if '_masking' in k])
    return masking_fns.get(name, masking_fns['tv'])


def prepare_experiment_config(config):
    """ Load all functions/classes/models requested in config to experiment config dict. """
    models = prepare_models(config['model'], device=config['device'])

    # Get preprocessors from base model
    # We can do that as they are the same for all datasets -- see HFLoRACLIPVisionModel
    if len(models['bases']) > 0 and hasattr(models['new'], 'train_preprocess'):
        if isinstance(config['dataset'], list):
            for dataset_config in config['dataset']:
                dataset_config['train_preprocess'] = models['new'].train_preprocess
                dataset_config['eval_preprocess'] = models['new'].val_preprocess
        else:
            config['dataset']['train_preprocess'] = models['new'].train_preprocess
            config['dataset']['eval_preprocess'] = models['new'].val_preprocess

    data = prepare_data(config['dataset'], device=config['device'])
    if config['eval_type'] == 'logits':
        if isinstance(data, list):
            dataset = data[-1]
        else:
            dataset = data

        if 'class_names' in dataset['test']:
            output_dim = len(dataset['test']['class_names'])
        else:
            output_dim = 1000

    else:
        output_dim = 512

    config['model']['output_dim'] = output_dim
    new_config = {
        'data': data,
        'models': models,
        'task_merge_config': config['task_merge_config'],
        'param_handler': prepare_param_handler(config['model'].get('ft_config', defaultdict()))
    }
    # Add outstanding elements
    for key in config:
        if key not in new_config:
            new_config[key] = config[key]
    return new_config


def get_config_from_name(name, device=None):
    """ Load config based on its name. """
    p = Path('configs', name)
    assert p.with_suffix('.py').exists(), f"Config file {p} does not exist."

    out = __import__('.'.join(p.parts[:-1]), fromlist=[p.stem]).__dict__[p.stem].config
    if device is None and 'device' not in out:
        out['device'] = 'cuda'
    elif device is not None:
        out['device'] = device
    return out


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_eval_args():
    parser = argparse.ArgumentParser(description='Run BIG function')

    parser.add_argument('--config', type=str, required=True, help='Config name')
    parser.add_argument('--representation', type=str, help='Representation type', choices=('vector', 'matrix_per_layer', 'svd'))
    parser.add_argument('--merge_space', type=str, help='Merge space', choices=('full', 'knots', 'core', 'core-vector', 'separate_a_b'))
    parser.add_argument('--merge_method', type=str, help='Merge method')
    parser.add_argument('--merging_type', type=str, help='used in vector representation')
    parser.add_argument('--scaling_coeffs', type=float, help='Scaling coefficients')
    parser.add_argument('--isotropize', type=str2bool, nargs='?', const=True, default=None,
                        help='Whether to isotropize the merged model')

    # Merging method specific parameters
    parser.add_argument('--topK', type=int, help='[TIES] Top K value')
    parser.add_argument('--dare_pruning_coeffs', type=float, help='[DARE] Pruning coefficients')
    parser.add_argument('--cart_pruning_rank', type=float, help='[CART] Pruning rank')

    parser.add_argument('--wandb', type=int, default=0, choices=(0, 1), help='Whether to log to wandb (1) or not (0)')
    parser.add_argument('--wandb_project', type=str, default='peft_merging', help='WandB project name')
    parser.add_argument('--wandb_entity', type=str, help='WandB entity name')

    parser.add_argument('--use_target_val', type=int, default=1, choices=(0, 1), help='Whether to use target validation for leave-one-out')

    return parser.parse_args()


def merge_args_into_task_merge_config(task_merge_config, args):
    """
    Overwrite or add entries in task_merge_config with non-None values from args (argparse.Namespace).
    """
    for key, value in vars(args).items():
        if key == "config":
            continue  # Don't overwrite config name in task_merge_config
        if value is not None:
            task_merge_config[key] = value
    return task_merge_config


##########################################################################################################################
#################################################### HELPER FUNCTIONS ####################################################
##########################################################################################################################

def set_seed(seed):
    """Set the seed for reproducibility."""
    #print("Setting Seed to {}".format(seed))
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def write_to_csv(results, csv_file):
    """Write results to a csv file."""
    if not os.path.exists(csv_file):
        # Create dir if necessary
        os.makedirs(os.path.dirname(csv_file), exist_ok=True)
        keys = list(results.keys())
        # Remove '_' and Capitalize first letter of every word
        keys = [str(key).replace('_', ' ').title() for key in keys]
        names = ','.join(keys)
        with open(csv_file, 'a') as f:
            f.write(f"{names}\n")

    csv_line = ','.join([str(i) for i in results.values()])
    with open(csv_file, 'a') as f:
        f.write(f"{csv_line}\n")


def get_device(model):
    """Get the device of the model."""
    return next(iter(model.parameters())).device


def load_clip_features(class_names, device):
    """Create CLIP target labels for class names. Return a normalized tensor of shape (num_classes, 512)."""
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in class_names]).to(device)
    model, preprocess = clip.load('ViT-B/32', device)
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)

    text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features


def create_heldout_split(dataset, fraction):  # root=dataset.root_og for most datasets
    root = dataset.root
    if hasattr(dataset, 'dataset'):
        val_set, test_set = train_test_split(dataset.dataset, test_size=fraction)
    else:
        val_set, test_set = train_test_split(dataset, test_size=fraction)
    val_set = dataset.__class__(root, train=dataset.train, transform=dataset.transform, base_set=val_set)
    test_set = dataset.__class__(root, train=dataset.train, transform=dataset.transform, base_set=test_set)
    return val_set, test_set


def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)


def load_model(model, save_path, model_device='cuda'):
    sd = torch.load(save_path, map_location=torch.device(model_device))
    model.load_state_dict(sd)
    return model


def mean_confidence_interval(data, confidence=0.95):
    """Get confidence interval of data"""
    # copied from: https://stackoverflow.com/questions/15033511/compute-a-confidence-interval-from-sample-data
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1) / 1.96
    return tuple(np.array([m, h]).round(5).tolist())


def get_clip_encodings(path):
    return torch.load(path)


def vector_to_state_dict(vector, state_dict, remove_keys=[]):
    # create a reference dict to define the order of the vector
    reference_dict = deepcopy(state_dict)
    for key in remove_keys:
        if key in reference_dict:
            del reference_dict[key]
    sorted_reference_dict = OrderedDict(sorted(reference_dict.items()))
    # create a shared state dict using the refence dict
    torch.nn.utils.vector_to_parameters(vector, sorted_reference_dict.values())
    # add back the encoder and decoder embedding weights.
    if "transformer.shared.weight" in sorted_reference_dict:
        for key in remove_keys:
            sorted_reference_dict[key] = sorted_reference_dict[
                "transformer.shared.weight"
            ]
    return sorted_reference_dict


def normalize(x, dim=0):
    min_values, _ = torch.min(x, dim=dim, keepdim=True)
    max_values, _ = torch.max(x, dim=dim, keepdim=True)
    y = (x - min_values) / (max_values - min_values)
    return y


def clamp(x, min_ratio=0, max_ratio=0):
    if len(x.size()) == 1:
        d = x.size(0)
        sorted_x, _ = torch.sort(x)
        min = sorted_x[int(d * min_ratio)]
        max = sorted_x[int(d * (1 - max_ratio) - 1)]
    else:
        d = x.size(1)
        sorted_x, _ = torch.sort(x, dim=1)
        min = sorted_x[:, int(d * min_ratio)].unsqueeze(1)
        max = sorted_x[:, int(d * (1 - max_ratio) - 1)].unsqueeze(1)
    clamped_x = torch.clamp(x, min, max)
    return clamped_x
