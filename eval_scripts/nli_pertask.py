import os
from copy import deepcopy
import time

import numpy as np
import torch

from task_merger import get_merge_handler
from utils import evaluate_logits, get_config_from_name, prepare_experiment_config, set_seed, parse_eval_args, merge_args_into_task_merge_config

# Set TOKENIZERS_PARALLELISM to true
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import transformers

transformers.utils.logging.set_verbosity(transformers.logging.ERROR)


def run_BIG_function(args):
    BIGSEED = 420

    print("Seed : ", BIGSEED)
    set_seed(BIGSEED)
    
    # Get config
    config_name = args.config
    print("Config name : ", config_name)

    TASK_HEADS_PATH = "data/llama-3.2-1B/heads.pt" if '1B' in config_name else "heads.pt"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    raw_config = get_config_from_name(config_name, device=device)
    print(raw_config['task_merge_config'])
    config = prepare_experiment_config(raw_config)
    config['task_merge_config'] = merge_args_into_task_merge_config(config['task_merge_config'], args)
    dataset_names = np.array([i['name'] for i in raw_config['dataset']])
    dataloaders = np.array([i for i in config['data']])
    mask_class = np.array([i['mask_class'] for i in config['dataset']])
    print(f"mask_class labels: {mask_class}")

    task_heads = torch.load(TASK_HEADS_PATH)

    finetuned_llama3_8b = {
        'snli': 92.49796416938111, 'mnli': 90.30820173204279, 'sick': 91.58173664900122, 'qnli': 94.48512585812358, 'rte': 89.85507246376812, 'scitail': 96.51928504233303, }
    finetuned_llama32_1b = {"mnli": 84.093, "snli": 88.578, "qnli": 89.725, 'sick': 90.216, 'rte': 78.986, 'scitail': 94.967}
    fine_tuned_acc = finetuned_llama3_8b if '8B' in config_name else finetuned_llama32_1b
    print(f'Finetuned Accs: {fine_tuned_acc}')

    def merge_and_eval(Merge, EVAL_SPLIT='val', instance_params=None):
        set_seed(BIGSEED)
        print("EVAL_SPLIT : ", EVAL_SPLIT)
        print(f'Search Run with: {instance_params}')
        all_results = deepcopy(instance_params)
        print('Creating Merge')

        Merge.set_scaling_coeffs(instance_params['scaling_coeffs'])
        config['task_merge_config'].update(instance_params)
        t0 = time.time()
        merged_model = Merge.merge(config['task_merge_config'])
        print(f"Time taken to merge: {time.time() - t0}")

        merged_model.config.pad_token_id = 128001
        merged_model.config.use_cache = False
        merged_model.config.pretraining_tp = 1

        print('Evaluate Merged Model on Each Dataset')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        avg_accuracy = 0.
        avg_norm_accuracy = 0.
        for i, loader_dict in enumerate(dataloaders):
            loader = loader_dict['test'][EVAL_SPLIT]
            with torch.no_grad():
                for name, param in merged_model.named_parameters():
                    # Inject task head into model
                    if 'modules_to_save' in name:
                        param.copy_(task_heads[dataset_names[i]])

            acc = evaluate_logits(merged_model, loader, device, mask_class[i])
            print(f"{dataset_names[i]} Normalized accuracy is {np.round((acc * 100)/ fine_tuned_acc[dataset_names[i]] *100, 3)}")
            print(f"{dataset_names[i]} accuracy is {np.round(acc * 100, 3)}")
            all_results[dataset_names[i]] = acc * 100
            all_results[dataset_names[i] + '_norm_acc'] = (acc * 100) / fine_tuned_acc[dataset_names[i]] * 100
            avg_accuracy += acc * 100
            avg_norm_accuracy += (acc * 100) / fine_tuned_acc[dataset_names[i]] * 100
        avg_accuracy /= len(dataloaders)
        avg_norm_accuracy /= len(dataloaders)
        print(f'Average Accuracy is {np.round(avg_accuracy, 3)}')
        print(f'Average Normalized Accuracy is {np.round(avg_norm_accuracy, 3)}')
        all_results['Average_acc'] = avg_accuracy
        all_results['Average_norm_acc'] = avg_norm_accuracy
        all_results.update(config['task_merge_config'])
        return all_results


    with torch.no_grad():
        lora_state_dicts = np.array([i for i in config['models']['bases']])
        MergeClass = get_merge_handler(config['task_merge_config']['representation'])
        Merge = MergeClass(
            lora_state_dicts,
            pretrained_model=config['models']['new'],
            param_handler=config['param_handler'],
            device=device,
            merge_config=config['task_merge_config'],
        )

        if config['task_merge_config']['ingredients_path'] is None or not os.path.exists(config['task_merge_config']['ingredients_path']):
            Merge.transform(config['task_merge_config'])

        test_result = merge_and_eval(Merge, EVAL_SPLIT='test', instance_params=config['task_merge_config'])
        datasets = ['snli', 'mnli', 'sick', 'qnli', 'rte', 'scitail']
        test_results = " & ".join([f"{np.round(test_result[dataset+'_norm_acc'], 2)}" for dataset in datasets]) + f" & {np.round(test_result['Average_norm_acc'], 2)} \\\\"
        print(f"Normalized Test results: {test_results}")
        print(test_result)


if __name__ == "__main__":
    args = parse_eval_args()
    run_BIG_function(args)
