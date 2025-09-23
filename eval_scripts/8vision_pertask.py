from copy import deepcopy

import numpy as np
import torch

from accuracies import get_vision_accuracies
from task_merger import get_merge_handler
from utils import evaluate_cliphead, get_clip_encodings, get_config_from_name, prepare_experiment_config, set_seed, parse_eval_args, merge_args_into_task_merge_config


def run_BIG_function(args):
    EVAL_SPLIT = 'test'
    BIGSEED = 420
    set_seed(BIGSEED)
    # Get config
    CONFIG_NAME = args.config

    print("Running with config: ", CONFIG_NAME)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    raw_config = get_config_from_name(CONFIG_NAME, device=device)
    # Get clip encodings
    all_clip_encodings = [get_clip_encodings(i['clip_encodings']) for i in raw_config['dataset']]
    config = prepare_experiment_config(raw_config)
    config['task_merge_config'] = merge_args_into_task_merge_config(config['task_merge_config'], args)
    dataset_names = [i['name'] for i in raw_config['dataset']]
    dataloaders = [i for i in config['data']]

    model_type = config['model']['base_type']
    rank = config['model']['ft_config'].get('r', None)
    peft_type = config['model']['ft_config'].get('type')
    fine_tuned_acc = get_vision_accuracies(model_type, peft_type=peft_type, rank=rank)

    print(raw_config['task_merge_config'])
    with torch.no_grad():
        all_results = deepcopy(config['task_merge_config'])
        print('Creating Merge')
        # iniitalize merging function
        models = np.array([i for i in config['models']['bases']])
        MergeClass = get_merge_handler(config['task_merge_config']['representation'])
        Merge = MergeClass(
            deepcopy(models),
            pretrained_model=deepcopy(config['models']['new']),
            param_handler=config['param_handler'],
            device=device,
            merge_config=config['task_merge_config'],
        )
        Merge.transform(config['task_merge_config'])
        # set task scaling coefficients
        Merge.set_scaling_coeffs(config['task_merge_config']['scaling_coeffs'])
        merged_model = Merge.merge(config['task_merge_config'])
        print('Evaluate Merged Model on Each Dataset')
        print("Using config: ", config['task_merge_config'])
        avg_accuracy = 0.
        avg_norm_accuracy = 0.
        for i, loader_dict in enumerate(dataloaders):
            loader = loader_dict['test'][EVAL_SPLIT]
            acc = evaluate_cliphead(merged_model.to(device), loader, class_vectors=all_clip_encodings[i].to(device))
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
        datasets = ['stanford_cars', 'dtd', 'eurosat', 'gtsrb', 'mnist', 'resisc45', 'sun397', 'svhn']
        test_results = " & ".join([f"{np.round(all_results[dataset+'_norm_acc'], 2)}" for dataset in datasets]) + f" & {np.round(all_results['Average_norm_acc'], 2)} \\\\"
        print(f"Normalized Test results: {test_results}")
        print('Finished!')


if __name__ == "__main__":
    args = parse_eval_args()
    run_BIG_function(args)
