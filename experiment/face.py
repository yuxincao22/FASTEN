from utils import lrange as L

config = dict(
    workers=24,
    print_freq=100,
    eval_freq=1000,

    evaluate=1, # 0 - train; 1- evaluate once; >1 train after evaluate
    
    batch_size=80,

    img_size=[96, 112],
    dataset='clip',
    nframes=5,
    drop_rate=0.2,

    test_path='./data/test_hifi.lst',

    benchmark='part',

    ignore_labels=dict(train=[], dev=[], test=[]),

    
    arch='mbv3_small,0.75',

    loss_type='ce',
    loss_weight=[1],
    ce_weight=[1,3],

    resume = './weights/weights_hifimask.pth'
)