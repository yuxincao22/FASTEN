import importlib
from math import gamma
from easydict import EasyDict as edict
from utils import lrange as L


_config = edict(
    # basic configs
    seed=47,
    workers=8,
    date=True,
    
    evaluate=1, # 0-no eval, 1-once, >1-eval first
    res_file=None,
    eval_verbose=False,
    eval_merge=False,
    optim_freq=1,
    print_freq=10,
    eval_freq=10,
    eval_name=['others', 'mask_ok', 'mask', 'mask_eye', 'pos_add'],
    topk=100,

    # dataset configs
    dataset='single_frame',
    nframes=1,
    max_skip=1,
    train_path='datasets/train.lst',
    dev_path='datasets/dev.lst',
    test_path='datasets/test.lst',
    benchmark='2he1',
    ignore_labels=dict(train=[], test=L(-108,-1)),
    key_name=None,

    # flow data
    downsample=2,
    with_speed_metric=True,
    padding_factor=4,

    # data transform 
    img_size=112,
    gamma_p=0.,
    scale_limit=0.15,
    jpeg_quality=500,
    jpeg_p=0.3,
    interpolation=1,
    interpolation_p=0.25,

    alb_aug_p=0.1,
    HorizontalFlip=False,
    RandomResizedCrop_large=0,
    
    # network configs
    arch='mbv3_small_trunc,0.25',
    # sometime no use
    internal_layers={'4': 'ft', '9': 'final'},
    # internal_indexes=[1, 3, 6, 12, 15],

    class_num=2,
    pretrained='',
    resume='',
    thrs=None,

    # training configs
    epochs=300,
    start_epoch=0,
    warmup_epoch=2,
    batch_size=2,
    epoch_size=0,
    # optimizer
    lr=0.0001,
    lr_mode='cosine',

    solver='adamw',
    momentum=0.9,
    beta=0.999,
    weight_decay=4e-4,
    bias_decay=0,
    param_group=True,
    param_freeze=False,
    bn_freeze=False,

    # ema
    use_ema=1,
    ema_decay=0.999,

    # loss configs
    loss_type='ce',
    loss_weight=[1],
    triplet_margin=0.5,
    ce_weight=None,
)


def struct_config(**config):
    def _print_format():
        io_strs = []
        for k, v in config.items():
            if isinstance(v, dict):
                io_str = f'{k}: '
                for subk, subv in v.items():
                    io_str += f'\n\t\t{subk}: {subv}'
            else:
                io_str = f'{k}: {v}'
            io_strs.append(io_str)
        return '\t'+'\n\t'.join(io_strs)

    def _extract(str_keys=''):
        cfg = edict()
        for key in str_keys.split(','):
            key = key.strip()
            if key == '': continue
            cfg.update({key: config.pop(key)})
        return cfg

    __config = config.copy()

    _data_cfg = _extract('dataset, train_path, dev_path, test_path, benchmark, ignore_labels')
    _trans_cfg = _extract('img_size, gamma_p, scale_limit, jpeg_p, jpeg_quality, interpolation, interpolation_p, alb_aug_p')

    _train_cfg = _extract('epochs, start_epoch, warmup_epoch, batch_size')
    _optim_cfg = _extract('lr, lr_mode, solver, momentum, beta, weight_decay, bias_decay, param_group, param_freeze')

    _net_cfg = _extract('arch, internal_layers, class_num, pretrained, resume')
    _loss_cfg = _extract('loss_type, loss_weight, triplet_margin, ce_weight')

    config.update(
        data_cfg=_data_cfg,
        trans_cfg=_trans_cfg,
        train_cfg=_train_cfg,
        optim_cfg=_optim_cfg,
        net_cfg=_net_cfg,
        loss_cfg=_loss_cfg,
    )
        
    __config.update(info=_print_format(), **config)

    return edict(__config)

def get_config(exp_tag):
    exp_module = f'.{exp_tag}'
    exp_config = importlib.import_module(exp_module, 'experiment')
    _config.update(**exp_config.config)
    return struct_config(**_config)