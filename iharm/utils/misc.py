import torch

from .log import logger


def get_dims_with_exclusion(dim, exclude=None):
    dims = list(range(dim))
    if exclude is not None:
        dims.remove(exclude)

    return dims


def save_checkpoint(net, optimizer, lr_scheduler, checkpoints_path, epoch, prefix='', verbose=True, multi_gpu=False):
    checkpoint_name = 'checkpoint.pth'

    if prefix:
        checkpoint_name = f'{prefix}_{checkpoint_name}'

    if not checkpoints_path.exists():
        checkpoints_path.mkdir(parents=True)

    checkpoint_path = checkpoints_path / checkpoint_name
    if verbose:
        logger.info(f'Save checkpoint to {str(checkpoint_path)}')

    torch.save({
        'model': net.module.state_dict() if multi_gpu else net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'epoch': epoch,
    }, str(checkpoint_path))

    return str(checkpoint_path)

def load_weights(model, path_to_weights, verbose=False):
    if verbose:
        logger.info(f'Load checkpoint from path: {path_to_weights}')

    checkpoint = torch.load(str(path_to_weights), map_location=torch.device('cpu'))
    if 'model' in checkpoint:
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model'], strict=False)
    else:
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)

    if len(missing_keys) > 0:
        print('Missing Keys: {}'.format(missing_keys))
    if len(unexpected_keys) > 0:
        print('Unexpected Keys: {}'.format(unexpected_keys))
