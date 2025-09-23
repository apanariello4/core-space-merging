from copy import deepcopy
import time
from itertools import product

import numpy as np
import torch

from accuracies import get_vision_accuracies
from task_merger import get_merge_handler
from utils import evaluate_cliphead, get_clip_encodings, get_config_from_name, merge_args_into_task_merge_config, parse_eval_args, prepare_experiment_config, set_seed


def get_all_combinations(param_dict):
    keys = list(param_dict.keys())
    values = list(param_dict.values())
    combinations = [dict(zip(keys, combo)) for combo in product(*values)]
    for combo in combinations:
        print(combo)
    return combinations


def run_BIG_function(args):
    EVAL_TEST = True
    # EVAL_SPLIT = 'test'
    EVAL_SPLIT = 'val'
    BIGSEED = 420
    EARLY_STOPPING_STEPS = 2

    print("Seed : ", BIGSEED)
    set_seed(BIGSEED)
    # Get config
    config_name = args.config
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    raw_config = get_config_from_name(config_name, device=device)
    # Get clip encodings
    all_clip_encodings = [get_clip_encodings(i['clip_encodings']) for i in raw_config['dataset']]
    config = prepare_experiment_config(raw_config)
    config['task_merge_config'] = merge_args_into_task_merge_config(config['task_merge_config'], args)
    dataset_names = [i['name'] for i in raw_config['dataset']]
    dataloaders = [i for i in config['data']]

    search_config = {
        'cart_pruning_rank': (0.04, 0.08, 0.16, 0.32),
        'cart_pruning_coeffs': np.arange(0.1, 10.0, step=0.4),
    }
    search_combinations = get_all_combinations(search_config)

    scaling_coeffs_search = np.arange(.1, 10.0, step=0.4)

    model_type = config['model']['base_type']
    rank = config['model']['ft_config'].get('r', None)
    peft_type = config['model']['ft_config'].get('type')
    fine_tuned_acc = get_vision_accuracies(model_type, peft_type=peft_type, rank=rank)

    print(f'Finetuned Accs: {fine_tuned_acc}')

    def merge_and_eval(EVAL_SPLIT='val', param_handler=None, instance_config=None):
        set_seed(BIGSEED)
        print("EVAL_SPLIT : ", EVAL_SPLIT)
        print(f'Search Run with: {instance_params}')
        all_results = deepcopy(instance_params)
        # iniitalize merging function
        print('Creating Merge')
        Merge = MergeClass(
            deepcopy(models),
            pretrained_model=deepcopy(config['models']['new']),
            param_handler=param_handler,
            device=device,
            merge_config=instance_config,
        )
        t1 = time.time()
        Merge.transform(instance_config)
        # set task scaling coefficients
        Merge.set_scaling_coeffs(instance_config['scaling_coeffs'])
        merged_model = Merge.merge(instance_config)
        t2 = time.time()
        print(f'Merging time: {t2 - t1:.2f} seconds')

        print('Evaluate Merged Model on Each Dataset')
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
        return all_results

    with torch.no_grad():
        print(search_combinations)
        models = np.array([i for i in config['models']['bases']])

        MergeClass = get_merge_handler(config['task_merge_config']['representation'])
        print(config['task_merge_config'])
        global_best_val_results = {'Average_norm_acc': 0.0}
        for search_combination in search_combinations:
            early_stopping = EARLY_STOPPING_STEPS
            local_best_val_results = {'Average_norm_acc': 0.0}
            for scaling_coeffs in scaling_coeffs_search:
                instance_params = deepcopy(search_combination)
                instance_params['scaling_coeffs'] = scaling_coeffs
                config['task_merge_config'].update(instance_params)
                print(config['task_merge_config'])
                all_results = merge_and_eval(
                    EVAL_SPLIT=EVAL_SPLIT,
                    param_handler=config['param_handler'],
                    instance_config=config['task_merge_config']
                )

                if all_results['Average_norm_acc'] >= global_best_val_results['Average_norm_acc']:
                    global_best_val_results = deepcopy(all_results)

                if (all_results['Average_norm_acc'] >= local_best_val_results['Average_norm_acc']):
                    local_best_val_results = deepcopy(all_results)
                    early_stopping = EARLY_STOPPING_STEPS
                else:
                    early_stopping -= 1
                    if (early_stopping == 0):
                        print(f"Early stopping for {search_combination=}")
                        break

        if EVAL_TEST:
            # Evaluate on the test set with the best topK and scaling co-efficient
            print("Best params :", global_best_val_results)
            for key in search_config.keys():
                instance_params.update({key: global_best_val_results[key]})
            instance_params['scaling_coeffs'] = global_best_val_results['scaling_coeffs']
            # test_result = merge_and_eval(Merge, EVAL_SPLIT = 'test', instance_params =instance_params)
            config['task_merge_config'].update(instance_params)
            test_result = merge_and_eval(
                EVAL_SPLIT='test',
                param_handler=config['param_handler'],
                instance_config=config['task_merge_config']
            )
            datasets = ['stanford_cars', 'dtd', 'eurosat', 'gtsrb', 'mnist', 'resisc45', 'sun397', 'svhn']
            test_results = " & ".join([f"{np.round(test_result[dataset+'_norm_acc'], 2):.2f}" for dataset in datasets]) + f" & {np.round(test_result['Average_norm_acc'], 2):.2f} \\\\"
            print(f"Normalized Test results: {test_results}")
            print(test_result)


if __name__ == "__main__":
    args = parse_eval_args()
    run_BIG_function(args)
