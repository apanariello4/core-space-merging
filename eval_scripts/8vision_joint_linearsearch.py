import os
import torch
from copy import deepcopy
import pickle

import numpy as np
from utils import evaluate_cliphead_joint, get_clip_encodings, get_config_from_name, prepare_experiment_config, set_seed, parse_eval_args, merge_args_into_task_merge_config
from collections import defaultdict
from task_merger import get_merge_handler


def run_BIG_function(args):
    EVAL_TEST = True
    BIGSEED = 420

    print("Seed : ", BIGSEED)
    set_seed(BIGSEED)
    # Get config
    CONFIG_NAME = args.config
    EARLY_STOPPING_STEPS = 2

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    raw_config = get_config_from_name(CONFIG_NAME, device=device)
    # Get clip encodings
    joint_stuff_dir = "./dataset/8vision_joint_components"
    print(f'Loading joint encodings from {joint_stuff_dir}')
    joint_encodings = get_clip_encodings(os.path.join(joint_stuff_dir, 'joint_head.pt'))
    dataset_mappers = pickle.load(open(os.path.join(joint_stuff_dir, 'joint_mappers.pkl'), 'rb'))
    config = prepare_experiment_config(raw_config)
    config['task_merge_config'] = merge_args_into_task_merge_config(config['task_merge_config'], args)
    dataset_names = [i['name'] for i in raw_config['dataset']]
    dataloaders = [i for i in config['data']]

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

    print(search_config)
    param_names, values = zip(*search_config.items())

    def merge_and_eval(Merge, EVAL_SPLIT='val', instance_params=None):
        set_seed(BIGSEED)
        print("EVAL_SPLIT : ", EVAL_SPLIT)
        print(f'Search Run with: {instance_params}')
        all_results = deepcopy(instance_params)
        print('Creating Merge')
        # set task scaling coefficients
        Merge.set_scaling_coeffs(instance_params['scaling_coeffs'])
        config['task_merge_config'].update(instance_params)
        merged_model = Merge.merge(config['task_merge_config'])

        print('Evaluate Merged Model on Each Dataset')
        joint_topk = defaultdict(lambda: 0)
        joint_total = 0.
        for i, loader_dict in enumerate(dataloaders):
            loader = loader_dict['test'][EVAL_SPLIT]

            label_map = torch.from_numpy(dataset_mappers[dataset_names[i]]['local2joint_map']).to(device)
            topk_counts, total, topk, confusion_matrix = evaluate_cliphead_joint(
                merged_model.to(device),
                loader,
                class_vectors=joint_encodings.to(device),
                aux_class_map=label_map,
            )

            joint_total += total
            for k, count in topk_counts.items():
                joint_topk[k] += count
            print(f"TopK for {dataset_names[i]}:")

            topk_prepared = {f"Top-{k}": f"{np.round(v * 100, 3)}" for k, v in topk.items()}
            print("\t".join([f"{k}: {v}" for k, v in topk_prepared.items()]))
            for k, v in topk_prepared.items():
                all_results[dataset_names[i] + f" {k}"] = v

        for k, v in joint_topk.items():
            all_results[f'Joint Top-{k}'] = np.round(v / joint_total * 100, 3)
        print("Joint TopK:")
        topk_prepared = {f"Top-{k}": f"{np.round(v / joint_total * 100, 3)}" for k, v in joint_topk.items()}
        print("\t".join([f"{k}: {v}" for k, v in topk_prepared.items()]))
        all_results.update(config['task_merge_config'])
        return all_results

    with torch.no_grad():

        print(search_config)
        param_names, values = zip(*search_config.items())

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
        print(config['task_merge_config'])
        # For linear search
        early_stopping = EARLY_STOPPING_STEPS
        for param in order_of_processing_params:
            best_val_results = {'Joint Top-3': 0.0}
            for value in search_config[param]:
                instance_params = deepcopy(default_params)
                instance_params[param] = value
                # pdb.set_trace()
                all_results = merge_and_eval(Merge, EVAL_SPLIT='val', instance_params=instance_params)
                if (all_results['Joint Top-3'] > best_val_results['Joint Top-3']):
                    best_val_results = deepcopy(all_results)
                    early_stopping = EARLY_STOPPING_STEPS
                else:
                    early_stopping -= 1
                    if (early_stopping <= 0):
                        break
            default_params[param] = best_val_results[param]
        # For grid search
        # best_val_results = {'Joint Top-3' : 0.0}
        # for bundle in product(*values):
        #     # pdb.set_trace()
        #     instance_params = dict(zip(param_names, bundle))
        #     all_results = merge_and_eval(Merge, EVAL_SPLIT = 'val', instance_params =instance_params)
        #     if (all_results['Joint Top-3'] > best_val_results['Joint Top-3']):
        #         best_val_results = deepcopy(all_results)

        if EVAL_TEST:
            # Evaluate on the test set with the best topK and scaling co-efficient
            print("Best params :", best_val_results)
            for key in search_config.keys():
                instance_params.update({key: best_val_results[key]})
            test_result = merge_and_eval(Merge, EVAL_SPLIT='test', instance_params=instance_params)
            print(test_result)


if __name__ == "__main__":
    args = parse_eval_args()
    run_BIG_function(args)
