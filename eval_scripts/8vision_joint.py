import os
import pickle
from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch

from task_merger import get_merge_handler
from utils import evaluate_cliphead_joint, get_clip_encodings, get_config_from_name, prepare_experiment_config, set_seed, parse_eval_args, merge_args_into_task_merge_config


def run_BIG_function(args):
    EVAL_TEST = True
    BIGSEED = 420

    print("Seed : ", BIGSEED)
    set_seed(BIGSEED)
    # Get config
    CONFIG_NAME = args.config

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    raw_config = get_config_from_name(CONFIG_NAME, device=device)
    # Get clip encodings
    joint_stuff_dir = "./dataset/8vision_joint_components"
    print(f'Loading joint encodings from {joint_stuff_dir}')
    joint_encodings = get_clip_encodings(os.path.join(joint_stuff_dir, 'joint_head.pt'))
    dataset_mappers = pickle.load(open(os.path.join(joint_stuff_dir, 'joint_mappers.pkl'), 'rb'))
    config = prepare_experiment_config(raw_config)
    config['task_merge_config'] = merge_args_into_task_merge_config(config['task_merge_config'], args)
    dataset_names = np.array([i['name'] for i in raw_config['dataset']])
    dataloaders = np.array([i for i in config['data']])

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
        # print(f'Joint accuracy over all datasets is {np.round(joint_accuracy * 100, 3)}')
        print("Joint TopK:")
        topk_prepared = {f"Top-{k}": f"{np.round(v / joint_total * 100, 3)}" for k, v in joint_topk.items()}
        print("\t".join([f"{k}: {v}" for k, v in topk_prepared.items()]))
        all_results.update(config['task_merge_config'])
        return all_results

    with torch.no_grad():
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

        if EVAL_TEST:
            print("Using config: ", config['task_merge_config'])
            test_result = merge_and_eval(
                Merge, EVAL_SPLIT='test',
                # instance_params =instance_params,
                instance_params=config['task_merge_config'],
            )
            print(test_result)


if __name__ == "__main__":
    args = parse_eval_args()
    run_BIG_function(args)
