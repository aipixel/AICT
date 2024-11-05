import torch
from pathlib import Path

from iharm.utils.misc import load_weights
from iharm.mconfigs import ALL_MCONFIGS
from iharm.utils.log import logger


def load_model(model_type, checkpoint_path, verbose=False):
    net = ALL_MCONFIGS[model_type]['model'](**ALL_MCONFIGS[model_type]['params'])
    # load_weights(net, checkpoint_path, verbose=verbose)

    logger.info(f'Load checkpoint from path: {checkpoint_path}')
    checkpoint = torch.load(str(checkpoint_path), map_location=torch.device('cpu'))
    state_dict = checkpoint['model']

    missing_keys, unexpected_keys = net.load_state_dict(state_dict, strict=False)
    if len(missing_keys) > 0:
        print('Missing Keys: {}'.format(missing_keys))
    if len(unexpected_keys) > 0:
        print('Unexpected Keys: {}'.format(unexpected_keys))

    return net


def find_checkpoint(weights_folder, checkpoint_name):
    weights_folder = Path(weights_folder)
    if ':' in checkpoint_name:
        model_name, checkpoint_name = checkpoint_name.split(':')
        models_candidates = [x for x in weights_folder.glob(f'{model_name}*') if x.is_dir()]
        assert len(models_candidates) == 1
        model_folder = models_candidates[0]
    else:
        model_folder = weights_folder

    if checkpoint_name.endswith('.pth'):
        if Path(checkpoint_name).exists():
            checkpoint_path = checkpoint_name
        else:
            checkpoint_path = weights_folder / checkpoint_name
    else:
        model_checkpoints = list(model_folder.rglob(f'{checkpoint_name}*.pth'))
        assert len(model_checkpoints) == 1
        checkpoint_path = model_checkpoints[0]
    return str(checkpoint_path)

