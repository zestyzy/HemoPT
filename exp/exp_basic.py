import os
import torch
from models.model_factory import get_model
from data_provider.data_factory import get_data


def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        total_params += params
    print(f"Total Trainable Params: {total_params}")
    return total_params


class Exp_Basic(object):
    def __init__(self, args):
        self.dataset, self.train_loader, self.test_loader, args.shapelist = get_data(args)
        _, _, self.test_loader_full, _ = get_data(args, full_mesh=True)
        requested_device = getattr(args, "device", "auto")
        if requested_device == "auto":
            requested_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(requested_device)
        args.device = str(self.device)
        if hasattr(self.dataset, "y_normalizer") and self.dataset.y_normalizer is not None:
            if hasattr(self.dataset.y_normalizer, "to"):
                self.dataset.y_normalizer.to(self.device)
        self.model = get_model(args).to(self.device)
        self.args = args
        print(self.args)
        print(self.model)
        count_parameters(self.model)

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
