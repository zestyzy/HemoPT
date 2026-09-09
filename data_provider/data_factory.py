from data_provider.data_loader import DrivAerML, NASA, AirCraft, DTCHull, Car_Crash
from data_provider.hemo_loader import HemoPT
from data_provider.hemo_vmr_cfd_loader import HemoVMRCFD
from data_provider.aneumo_cfd_loader import AneumoCFD
from data_provider.vascular_pretrain_loader import VascularPretrain
from data_provider.vmr_cfd_loader import VMRCFD


def get_data(args, full_mesh=False):
    data_dict = {
        'DrivAerML': DrivAerML,
        'NASA': NASA,
        'AirCraft': AirCraft,
        'DTCHull': DTCHull,
        'Car_Crash': Car_Crash,
        'HemoPT': HemoPT,
        'HemoVMRCFD': HemoVMRCFD,
        'AneumoCFD': AneumoCFD,
        'VascularPretrain': VascularPretrain,
        'VMRCFD': VMRCFD,
    }
    dataset = data_dict[args.loader](args)
    train_loader, test_loader, shapelist = dataset.get_loader(full_mesh)
    return dataset, train_loader, test_loader, shapelist
