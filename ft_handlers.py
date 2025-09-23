import torch.nn as nn
from collections import defaultdict, OrderedDict

"""
True base_model.model.classifier.original_module.dense.weight
True base_model.model.classifier.original_module.dense.bias
True base_model.model.classifier.original_module.out_proj.weight
True base_model.model.classifier.original_module.out_proj.bias

"""


class LoRAHandler():
    def __init__(self, state_dict):
        self.state_dict = state_dict

    def get_ft_parameters(self):
        layer2lora_parameters = defaultdict(lambda: dict())
        sd = self.state_dict
        if not hasattr(sd, 'items'):
            # peft > 0.3.0 fix
            sd = sd.state_dict()

        for key, val in sd.items():
            if '.lora_A.default' in key:
                base_name = key.replace('.lora_A.default', '')
                layer2lora_parameters[base_name]['A'] = val
            elif '.lora_A' in key:
                base_name = key.replace('.lora_A', '')
                layer2lora_parameters[base_name]['A'] = val
            elif '.lora_B.default' in key:
                base_name = key.replace('.lora_B.default', '')
                layer2lora_parameters[base_name]['B'] = val
            elif '.lora_B' in key:
                base_name = key.replace('.lora_B', '')
                layer2lora_parameters[base_name]['B'] = val

        task_parameters = {}
        for name, key2val in layer2lora_parameters.items():
            # A: [r, I]. B: [O, r]. BxA: [O,r]x[r,I]:[O,I].
            task_parameters[name] = (key2val['B'] @ key2val['A'])
        return OrderedDict(sorted(task_parameters.items()))

    def get_ft_ab_parameters(self):
        layer2lora_parameters = defaultdict(lambda: dict())
        sd = self.state_dict
        for key, val in sd.items():
            if '.lora_A.default' in key:
                base_name = key.replace('.lora_A.default', '')
                layer2lora_parameters[base_name]['A'] = val
            elif '.lora_A' in key:
                base_name = key.replace('.lora_A', '')
                layer2lora_parameters[base_name]['A'] = val
            elif '.lora_B.default' in key:
                base_name = key.replace('.lora_B.default', '')
                layer2lora_parameters[base_name]['B'] = val
            elif '.lora_B' in key:
                base_name = key.replace('.lora_B', '')
                layer2lora_parameters[base_name]['B'] = val

        task_parameters = {}
        for name, key2val in layer2lora_parameters.items():
            task_parameters[name] = (key2val['A'], key2val['B'])
        return OrderedDict(sorted(task_parameters.items()))


class VeRAHandler():
    def __init__(self, state_dict):
        self.state_dict = state_dict

    def get_ft_parameters(self):
        layer2vera_parameters = self.extract_layer_parameters()
        task_parameters = {}
        for name, key2val in layer2vera_parameters.items():
            task_parameters[name + '.weight'] = (key2val['lambda_B'].unsqueeze(-1) * key2val['B']) @ (key2val['lambda_D'].unsqueeze(-1) * key2val['A'])
        return OrderedDict(sorted(task_parameters.items()))

    def get_ft_ab_parameters(self):
        layer2vera_parameters = self.extract_layer_parameters()
        task_parameters = {}
        for name, key2val in layer2vera_parameters.items():
            task_parameters[name + '.weight'] = ((key2val['lambda_D'].unsqueeze(-1) * key2val['A']), (key2val['lambda_B'].unsqueeze(-1) * key2val['B']))
        return OrderedDict(sorted(task_parameters.items()))

    def extract_layer_parameters(self):
        layer2vera_parameters = defaultdict(lambda: dict())
        sd = self.state_dict
        if not hasattr(sd, 'items'):
            # peft > 0.3.0 fix
            sd = sd.state_dict()

        for key, val in sd.items():
            if '.vera_lambda_d.default' in key:
                base_name = key.replace('.vera_lambda_d.default', '')
                layer2vera_parameters[base_name]['lambda_D'] = val
            elif '.vera_lambda_d' in key:
                base_name = key.replace('.vera_lambda_d', '')
                layer2vera_parameters[base_name]['lambda_D'] = val
            elif '.vera_lambda_b.default' in key:
                base_name = key.replace('.vera_lambda_b.default', '')
                layer2vera_parameters[base_name]['lambda_B'] = val
            elif '.vera_lambda_b' in key:
                base_name = key.replace('.vera_lambda_b', '')
                layer2vera_parameters[base_name]['lambda_B'] = val
            elif '.vera_A.default' in key:
                base_name = key.replace('.vera_A.default', '')
                layer2vera_parameters[base_name]['A'] = val
            elif '.vera_A' in key:
                base_name = key.replace('.vera_A', '')
                layer2vera_parameters[base_name]['A'] = val
            elif '.vera_B.default' in key:
                base_name = key.replace('.vera_B.default', '')
                layer2vera_parameters[base_name]['B'] = val
            elif '.vera_B' in key:
                base_name = key.replace('.vera_B', '')
                layer2vera_parameters[base_name]['B'] = val
        layer2vera_parameters.pop('vision_model.base_model', None)
        return layer2vera_parameters


class FFTHandler(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def get_ft_parameters(self):
        return OrderedDict(sorted(self.base_model.state_dict().items()))

    def get_final_model(self, **kwargs):
        return self.base_model


class GeneralHandler(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def get_ft_parameters(self):
        return OrderedDict(sorted(self.base_model.state_dict().items()))

    def get_final_model(self, **kwargs):
        return self.base_model
