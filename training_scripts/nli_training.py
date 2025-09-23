import os
from pathlib import Path

# Set TOKENIZERS_PARALLELISM to true
os.environ["TOKENIZERS_PARALLELISM"] = "true"

from huggingface_hub import login

# Get the token from environment variables
# token = os.getenv('HUGGINGFACE_TOKEN')
with open("hf", "r") as f:
    token = f.read().strip()
login(token=token)

import torch
import transformers
from peft import LoraConfig, get_peft_model

from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup

from utils import evaluate_logits

transformers.utils.logging.set_verbosity(transformers.logging.ERROR)


def grab_nli_loader_fns(name):
    """ Returns the dataset loader functions for the specified NLI dataset """
    if name == 'snli':
        from dataset.snli import prepare_test_loaders, prepare_train_loaders
    elif name == 'mnli':
        from dataset.mnli import prepare_test_loaders, prepare_train_loaders
    elif name == 'sick':
        from dataset.sick import prepare_test_loaders, prepare_train_loaders
    elif name == 'qnli':
        from dataset.qnli import prepare_test_loaders, prepare_train_loaders
    elif name == 'rte':
        from dataset.rte import prepare_test_loaders, prepare_train_loaders
    elif name == 'scitail':
        from dataset.scitail import prepare_test_loaders, prepare_train_loaders
    else:
        raise NotImplementedError(name)

    return prepare_train_loaders, prepare_test_loaders


def grab_nli_dataset_configs(name):
    """ Returns the dataset config for the specified NLI dataset """
    if name == 'snli':
        from dataset.configs import snli as base_config
    elif name == 'mnli':
        from dataset.configs import mnli as base_config
    elif name == 'sick':
        from dataset.configs import sick as base_config
    elif name == 'qnli':
        from dataset.configs import qnli as base_config
    elif name == 'rte':
        from dataset.configs import rte as base_config
    elif name == 'scitail':
        from dataset.configs import scitail as base_config
    else:
        raise NotImplementedError(name)
    return base_config


"""
Original Label info:
snli: 0 - entailment, 1 - neutral, 2 - contradiction
mnli: 0 - entailment, 1 - neutral, 2 - contradiction
sick: 0 - entailment, 1 - neutral, 2 - contradiction
qnli: 0 - entailment, 1 - non-entailment
rte : 0 - entailment, 1 - not-entailment
scitail : entails and neutral
"""


def main(args):

    # ----------------- Edit from here -----------------#
    CACHE_DIR = "data"
    MODEL_SAVE_DIR = ""
    MAX_NUM_EPOCHS = 10
    EVAL_AFTER_STEPS = 4000
    TASK = args.task
    PREPARE_TRAIN_LOADERS, PREPARE_TEST_LOADERS = grab_nli_loader_fns(TASK)
    DATASET_CONFIG = grab_nli_dataset_configs(TASK)
    LR = 3e-5
    BATCH_SIZE = 1
    NUM_WORKERS = 1

    DATASET_CONFIG['batch_size'] = BATCH_SIZE
    DATASET_CONFIG['num_workers'] = NUM_WORKERS
    # MODEL_NAME_OR_PATH = "meta-llama/Meta-Llama-3-8B"
    MODEL_NAME_OR_PATH = args.model
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    peft_config = LoraConfig(task_type="SEQ_CLS", inference_mode=False, r=16, lora_alpha=16, lora_dropout=0.1, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])

    cache_dir = CACHE_DIR

    train_dataloader = PREPARE_TRAIN_LOADERS(DATASET_CONFIG)['full']
    val_dataloader = PREPARE_TEST_LOADERS(DATASET_CONFIG)['val']
    test_dataloader = PREPARE_TEST_LOADERS(DATASET_CONFIG)['test']

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME_OR_PATH, return_dict=True, cache_dir=cache_dir, num_labels=3)
    model = get_peft_model(model, peft_config)

    mask_class = DATASET_CONFIG['mask_class']
    print(mask_class)
    print(model.print_trainable_parameters())

    tokenizer = PREPARE_TRAIN_LOADERS(DATASET_CONFIG)['tokenizer']
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    optimizer = AdamW(params=model.parameters(), lr=LR)
    # Instantiate scheduler
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0.06 * (len(train_dataloader) * MAX_NUM_EPOCHS),
        num_training_steps=(len(train_dataloader) * MAX_NUM_EPOCHS),
    )

    criterion = CrossEntropyLoss()
    model = model.to(DEVICE)

    print(f"LoRA Task is: {TASK}")
    total_steps = 0
    early_stopping = 3
    max_acc = 0
    for epoch in range(MAX_NUM_EPOCHS):
        model.train()
        for batch in tqdm(train_dataloader, desc=f"Training Ep. {epoch + 1}", total=len(train_dataloader)):
            total_steps += 1
            batch.to(DEVICE)
            outputs = model(**batch)
            if mask_class is not None:
                outputs.logits[:, mask_class] = -1e10
            loss = criterion(outputs.logits, batch.labels)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            if total_steps % EVAL_AFTER_STEPS == 0:
                model.eval()
                acc = evaluate_logits(model, val_dataloader, DEVICE, mask_class=mask_class)
                print(f"epoch {epoch+1} val acc : {acc:.3%}")
                if acc > max_acc:
                    torch.save(model.state_dict(), Path(MODEL_SAVE_DIR, f'{TASK}.pt'))
                    max_acc = acc
                    early_stopping = 3
                else:
                    early_stopping -= 1
                if early_stopping == 0:
                    print("Early stopping")
                    break
        if early_stopping == 0:
            break
    print("Training finished")
    print(f"Best val acc : {max_acc:.3%}")
    model.eval()

    model.load_state_dict(torch.load(Path(MODEL_SAVE_DIR, f'{TASK}.pt'), map_location=torch.device(DEVICE)))
    acc = evaluate_logits(model, test_dataloader, DEVICE, mask_class=mask_class)
    print(f"test acc : {acc:.3%}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train NLI model")
    parser.add_argument("--task", type=str, help="Task to train model on", choices=("snli", "mnli", "sick", "qnli", "rte", "scitail"))
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B", help="Model to train name or path")
    args = parser.parse_args()

    main(args)
