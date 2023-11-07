import warnings
warnings.filterwarnings("ignore")
import argparse
import dist
import torch
from torch.utils.data import DataLoader, DistributedSampler, SequentialSampler
import torchvision.transforms as transforms
from experiment.config import get_config
from data.dataset import ClipDataset
from data.dataset import load_clip_list
from metrics.roc import evaluate_roc
from models.fasten import Fasten


def get_args_parser():
    parser = argparse.ArgumentParser(description='FASTEN', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--exp_tag', default='face',
                        help='experiment tag for storage folder' )    

    # distribute
    parser.add_argument('--world_size', default=-1, type=int, help='number of distributed process')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    args = parser.parse_args()
    cfgs = get_config(args.exp_tag)
    cfgs.update(args.__dict__)

    return cfgs

args = get_args_parser()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_data_loader():
    anno_parser = load_clip_list

    input_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    norm_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    test_transform = dict(
        transform=input_transform,
        norm_transform=norm_transform,
    )
    return anno_parser, test_transform    

def build_net():
    model = Fasten('mobilenet,0.75', args.nframes, 2, pretrained='models/pretrained/mobilenetv3-small-0.75-86c972c3.pth', drop_rate=args.drop_rate)
    if args.net_cfg.resume != None and args.net_cfg.resume != '':
        resume = args.net_cfg.resume
        print(f"=> Using weights from '{resume}'")
        net_data = torch.load(resume, map_location='cpu')
        try:
            model.load_state_dict(net_data['state_dict'],strict=False)
        except:
            model.load_state_dict(net_data,strict=False)
    return model.to(device)

def main():
    dist.init_distributed_mode(args)

    model = build_net()
    anno_parser, test_transform = set_data_loader()
    print(f'=> load test datasets')
    test_set = ClipDataset(args.test_path, anno_parser, args.nframes, args.max_skip, ignore_labels=args.ignore_labels['test'], transform=test_transform, phase='test')
    test_sampler = DistributedSampler(test_set, shuffle = False) if args.distributed else SequentialSampler(test_set)
    test_data = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.workers, drop_last=False, sampler=test_sampler)

    print('=> Evaluating')
    evaluate_roc(test_data, model)

if __name__ == '__main__':
    main()
