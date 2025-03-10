from iharm.model.base import AICT
                    
BMCONFIGS = {

########################
###### ViT-based #######
########################
    'ViT_aict': {
        'model': AICT,
        'params': {'backbone_type': 'ViT',
                   'input_normalization': {'mean': [0, 0, 0], 'std': [1, 1, 1]},
                   'linear_num': 8,
                   'clamp': True, 'color_space': 'RGB', 'use_attn': False
                   },
        'data': {'color_space': 'RGB'}
    }
}