def get_vision_accuracies(model, rank, peft_type, n_tasks=8, dataset_names=None):
    if isinstance(rank, list):
        assert dataset_names is not None, "dataset_names must be provided when rank is a list"
        return {dataset_names[i]: get_vision_accuracies(model, r, peft_type)[dataset_names[i]] for i, r in enumerate(rank)}
    if model == "openai/clip-vit-large-patch14" and rank == 16 and peft_type == "lora":
        return {
            'stanford_cars': 99.76682729675113,
            'dtd': 70.0531914893617,
            'eurosat': 98.59259259259259,
            'gtsrb': 97.19912905779889,
            'mnist': 99.525,
            'resisc45': 95.69841269841269,
            'sun397': 79.59697732997482,
            'svhn': 97.72399884759435,
        }
    if model == "openai/clip-vit-base-patch32" and rank == 16 and peft_type == "lora":
        return {
            'stanford_cars': 74.0,
            'dtd': 58.3,
            'eurosat': 99.0,
            'gtsrb': 92.7,
            'mnist': 99.3,
            'resisc45': 88.4,
            'sun397': 64.5,
            'svhn': 96.2
        }
    if model == "openai/clip-vit-base-patch32" and rank == 64 and peft_type == "lora":
        return {
            'stanford_cars': 99.6269236748018,
            'dtd': 68.03191489361702,
            'eurosat': 97.33333333333334,
            'gtsrb': 98.00079176563737,
            'mnist': 99.3625,
            'resisc45': 93.85714285714286,
            'sun397': 70.85642317380353,
            'svhn': 96.24987995774512
        }
    if model == "openai/clip-vit-base-patch32" and rank == 256 and peft_type == "lora":
        return {
            'stanford_cars': 99.73573760298461,
            'dtd': 68.13829787234043,
            'eurosat': 98.37037037037037,
            'gtsrb': 98.3174980205859,
            'mnist': 99.325,
            'resisc45': 93.96825396825397,
            'sun397': 72.40554156171285,
            'svhn': 96.60040334197637
        }
    if model == "openai/clip-vit-base-patch32" and rank == 16 and peft_type == "vera":
        return {
            'stanford_cars': 62.79596977329975,
            'dtd': 57.07446808510638,
            'eurosat': 96.55555555555555,
            'gtsrb': 90.85510688836105,
            'mnist': 98.6,
            'resisc45': 88.50793650793651,
            'sun397': 62.79596977329975,
            'svhn': 93.10957457024873
        }
    if model == "openai/clip-vit-base-patch32" and rank == 1 and peft_type == "lora":
        return {
            'stanford_cars': 99.8911860718172,
            'dtd': 65.95744680851063,
            'eurosat': 97.74074074074074,
            'gtsrb': 95.37806809184481,
            'mnist': 99.0625,
            'resisc45': 92.23809523809524,
            'sun397': 66.40428211586902,
            'svhn': 95.70248727552099,
        }
