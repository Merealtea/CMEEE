from typing_extensions import runtime
from transformers import Trainer
from torch import nn
import torch

def get_parameter_names(model, forbidden_layer_types):
    """
    Returns the names of the model parameters that are not inside a forbidden layer.
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result


class Trainer_lr_decay(Trainer):
    def __init__(self, 
        model = None,
        args = None,
        data_collator = None,
        train_dataset = None,
        eval_dataset = None,
        tokenizer = None,
        model_init = None,
        compute_metrics = None,
        callbacks  = None,
        optimizers = (None, None),
        lr_decay_rate = None) :
        super(Trainer_lr_decay, self).__init__(model,args ,
                                                data_collator,
                                                train_dataset,
                                                eval_dataset,
                                                tokenizer,
                                                model_init,
                                                compute_metrics,
                                                callbacks,
                                                optimizers)
        # if Trainer.__version__ != '4.11.0':
        #     raise RuntimeError(f"要使用Trainer版本为4.11.0,当前版本为{Trainer.__version__}")
        self.lr_decay_rate = lr_decay_rate
        """丑是丑了点，能用就行
        """

    def create_optimizer(self):
        """
        Override
        Setup the optimizer.
        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, [nn.LayerNorm])
            decay_parameters = [name for name in decay_parameters if "bias" not in name]

            # 先找到逐层的名字，后面按照层的类型设置lr
            layer_names = []
            for name, _ in self.model.named_parameters():
                layer_names.append(name)

            # 越底层的layer学习率要越大
            layer_names.reverse()

            optimizer_grouped_parameters = []
            prev_name = layer_names[0].split('.')[0]
            lr = self.args.learning_rate

            for name in layer_names:
                cur_name = name.split('.')[0]

                if cur_name != prev_name:
                    lr *= self.lr_decay_rate
                prev_name = cur_name
                
                if name in decay_parameters:
                    weight_decay = self.args.weight_decay
                else:
                    weight_decay = 0.0
                optimizer_grouped_parameters += [{'params': 
                                    [p for n, p in opt_model.named_parameters() if n == name and p.requires_grad], 
                                'lr': lr,
                                "weight_decay": weight_decay,
                                }]

            optimizer_kwargs = {
                    "betas": (self.args.adam_beta1, self.args.adam_beta2),
                    "eps": self.args.adam_epsilon,
                }

            self.optimizer = torch.optim.AdamW(
                    params=optimizer_grouped_parameters,
                    **optimizer_kwargs)
        return self.optimizer

# def create_lr_decay_optimizer(trainer, lr_decay):
#     layer_names = []
#     for name, _ in trainer.model.named_parameters():
#         layer_names.append(name)

#     # 越底层的layer学习率要越大
#     layer_names.reverse()

#     parameters = []

#     prev_name = layer_names[0].split('.')[0]
#     lr = trainer.lr

#     for name in layer_names:
#         cur_name = name.split('.')[0]

#         if cur_name != prev_name:
#             lr *= self.lr_decay_rate
#         prev_name = cur_name
        
#         parameters += [{'params': 
#                             [p for n, p in model.named_parameters() if n == name and p.requires_grad], 
#                         'lr': lr}]
#     optim = torch.optim.AdamW(parameters,
#                             weight_decay= weight_decay_rate)
