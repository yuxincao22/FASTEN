import os
import torch
import torch.distributed as dist


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.rank = int(os.environ['RANK'])
        args.gpu_id = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID']) # global rank
        args.gpu_id = args.rank % torch.cuda.device_count() # gpu num per device
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url))

    torch.cuda.set_device(args.gpu_id)
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    setup_for_distributed(args.rank == 0)

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_dist_avail_and_initialized():
    if dist.is_available() and dist.is_initialized():
        return True
    return False

def is_main_process():
    return get_rank() == 0
