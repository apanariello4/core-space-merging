import time
from copy import deepcopy

import numpy as np
import torch
import wandb

from accuracies import get_vision_accuracies
from task_merger import get_merge_handler
from utils import (
    evaluate_cliphead,
    get_clip_encodings,
    get_config_from_name,
    merge_args_into_task_merge_config,
    parse_eval_args,
    prepare_experiment_config,
    set_seed,
)


def run_BIG_function(args):
    EVAL_TEST = True
    # EVAL_SPLIT = 'test'
    EVAL_SPLIT = 'val'
    BIGSEED = 420

    print("Seed : ", BIGSEED)
    set_seed(BIGSEED)

    # Get config
    config_name = args.config
    print("Config name : ", config_name)
    EARLY_STOPPING_STEPS = 2

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    raw_config = get_config_from_name(config_name, device=device)
    if args.wandb:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                   name=f"{config_name}_{args.merge_method}_{args.merge_space}_{args.representation}",
                   config={'raw_config': raw_config, 'args': vars(args)})
    # Get clip encodings
    all_clip_encodings = [get_clip_encodings(i['clip_encodings']) for i in raw_config['dataset']]
    config = prepare_experiment_config(raw_config)
    config['task_merge_config'] = merge_args_into_task_merge_config(config['task_merge_config'], args)
    dataset_names = [i['name'] for i in raw_config['dataset']]
    dataloaders = [i for i in config['data']]

    # Parameters are tuned in the order specified in search_config
    default_params = {'scaling_coeffs': .6,
                      'topK': 30,
                      'cart_pruning_rank': 0.04,
                      'dare_pruning_coeffs': 1e-5,
                      }  # Default config
    order_of_processing_params = [
        'scaling_coeffs',
    ]
    search_config = {
        'scaling_coeffs': np.arange(0.1, 10, step=0.1),
        'topK': (np.arange(1, 11, step=1) * 10)[::-1],
        'dare_pruning_coeffs': [0.99, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 1e-5][::-1],
        'cart_pruning_rank': [0.04, 0.08, 0.16, 0.32]
    }

    if 'dare' in config['task_merge_config']['merge_method']:
        order_of_processing_params.append('dare_pruning_coeffs')
    if 'ties' in config['task_merge_config']['merge_method']:
        order_of_processing_params.append('topK')
    if 'cart' in config['task_merge_config']['merge_method']:
        order_of_processing_params.append('cart_pruning_rank')

    model_type = config['model']['base_type']
    rank = config['model']['ft_config'].get('r', None)
    peft_type = config['model']['ft_config'].get('type')
    fine_tuned_acc = get_vision_accuracies(model_type, peft_type=peft_type, rank=rank, dataset_names=dataset_names)

    print(f'Finetuned Accs: {fine_tuned_acc}')

    def merge_and_eval(merger, EVAL_SPLIT='val', instance_config=None):
        set_seed(BIGSEED)
        print("EVAL_SPLIT : ", EVAL_SPLIT)
        print(f'Search Run with: {instance_params}')
        all_results = deepcopy(instance_params)
        # initialize merging function
        print('Creating Merge')
        t1 = time.time()
        merger.transform(instance_config)
        # set task scaling coefficients
        merger.set_scaling_coeffs(instance_config['scaling_coeffs'])
        merged_model = merger.merge(instance_config)
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
        # Log the merge evaluation results to wandb
        if args.wandb:
            wandb.log({**all_results, "params": instance_params})
        return all_results

    with torch.no_grad():
        print(search_config)
        models = np.array([i for i in config['models']['bases']])

        MergeClass = get_merge_handler(config['task_merge_config']['representation'])
        merger = MergeClass(
            deepcopy(models),
            pretrained_model=deepcopy(config['models']['new']),
            param_handler=config['param_handler'],
            device=device,
            merge_config=config['task_merge_config'],
        )
        print(config['task_merge_config'])
        early_stopping = EARLY_STOPPING_STEPS
        for param in order_of_processing_params:
            best_val_results = {'Average_norm_acc': 0.0}
            for value in search_config[param]:
                instance_params = deepcopy(default_params)
                instance_params[param] = value
                config['task_merge_config'].update(instance_params)
                all_results = merge_and_eval(
                    merger=merger,
                    EVAL_SPLIT=EVAL_SPLIT,
                    instance_config=config['task_merge_config']
                )
                if (all_results['Average_norm_acc'] >= best_val_results['Average_norm_acc']):
                    best_val_results = deepcopy(all_results)
                    early_stopping = EARLY_STOPPING_STEPS
                else:
                    early_stopping -= 1
                    if (early_stopping == 0):
                        print("Early stopping")
                        break
            default_params[param] = best_val_results[param]

        if EVAL_TEST:
            # Evaluate on the test set with the best topK and scaling co-efficient
            print("Best params :", best_val_results)
            for key in search_config.keys():
                instance_params.update({key: best_val_results[key]})
            config['task_merge_config'].update(instance_params)
            test_result = merge_and_eval(
                merger=merger,
                EVAL_SPLIT='test',
                instance_config=config['task_merge_config']
            )
            datasets = ['stanford_cars', 'dtd', 'eurosat', 'gtsrb', 'mnist', 'resisc45', 'sun397', 'svhn']
            test_results = " & ".join([f"{np.round(test_result[dataset+'_norm_acc'], 2):.2f}" for dataset in datasets]) + f" & {np.round(test_result['Average_norm_acc'], 2):.2f} \\\\"
            print(f"Normalized Test results: {test_results}")
            print(test_result)
            # Save results to results.txt
            with open("results.txt", "a") as f:
                f.write(f"Args: {vars(args)}\n")
                f.write(f"Normalized Test results: {test_results}\n")
                f.write(f"Test result dict: {test_result}\n")
                f.write(f"Best parameters: {instance_params}\n\n")
            # Log final test results to wandb
            if args.wandb:

                wandb.log({"final_test": test_result, "best_parameters": instance_params})
    if args.wandb:
        # Finish the wandb run
        wandb.finish()


if __name__ == "__main__":
    args = parse_eval_args()
    run_BIG_function(args)
