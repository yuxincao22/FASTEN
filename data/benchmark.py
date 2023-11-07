from utils import lrange as L

class_dict_3dmask = {
    1: 'live',
    0: 'mask',
}

_benchmarks = {
    'part': dict(
        name=class_dict_3dmask,
        pos_labels=[1],
        key_name='mask',
        groups=dict(
            mask=dict(neg_labels=[0]), 
        )
    ),
}

def get_benchmark(version):
    return _benchmarks[version]