from easydict import EasyDict as edict


def add_default_sampler(batch_sampler):
    keys = ['min_scale','max_scale','min_aspect_ratio','max_aspect_ratio',
        'min_jaccard_overlap','max_jaccard_overlap','max_trials','max_sample']
    for i in range(1,len(batch_sampler)):
        for key in keys:
            if batch_sampler[i].get(key) == None:
                default_val = batch_sampler[0][key]
                batch_sampler[i][key] = default_val
                
def get_batch_sampler():
     # the first item is the default values
    batch_sampler = [
        {
            'max_sample':1,
            'max_trials':1,
            'min_scale':1,
            'max_scale':1,
            'min_aspect_ratio':1,
            'max_aspect_ratio':1,
            'min_jaccard_overlap':0,
            'max_jaccard_overlap':1,
        },
        {
            'max_trials': 1,
            'max_sample': 1,
        },
        {
            'min_scale': 0.3,
            'max_scale': 1.0,
            'min_aspect_ratio': 0.5,
            'max_aspect_ratio': 2.0,
            'min_jaccard_overlap': 0.1,
            'max_trials': 50,
            'max_sample': 1,
        },
        {
            'min_scale': 0.3,
            'max_scale': 1.0,
            'min_aspect_ratio': 0.5,
            'max_aspect_ratio': 2.0,
            'min_jaccard_overlap': 0.3,
            'max_trials': 50,
            'max_sample': 1,
        },
        {
            'min_scale': 0.3,
            'max_scale': 1.0,
            'min_aspect_ratio': 0.5,
            'max_aspect_ratio': 2.0,
            'min_jaccard_overlap': 0.5,
            'max_trials': 50,
            'max_sample': 1,
        },
        {
            'min_scale': 0.3,
            'max_scale': 1.0,
            'min_aspect_ratio': 0.5,
            'max_aspect_ratio': 2.0,
            'min_jaccard_overlap': 0.7,
            'max_trials': 50,
            'max_sample': 1,
        },
        {
            'min_scale': 0.3,
            'max_scale': 1.0,
            'min_aspect_ratio': 0.5,
            'max_aspect_ratio': 2.0,
            'min_jaccard_overlap': 0.9,
            'max_trials': 50,
            'max_sample': 1,
        },
        {
            'min_scale': 0.3,
            'max_scale': 1.0,
            'min_aspect_ratio': 0.5,
            'max_aspect_ratio': 2.0,
            'max_jaccard_overlap': 1.0,
            'max_trials': 50,
            'max_sample': 1,
        },
    ]
    
    add_default_sampler(batch_sampler)
    
    return batch_sampler


    
    
def config():
    cfg = edict()
    cfg.batch_sampler = get_batch_sampler()
    print cfg.batch_sampler
    
    cfg.expand_param = edict()
    cfg.expand_param.prob = 1.0
    cfg.expand_param.min_expand_ratio = 1.0
    cfg.expand_param.max_expand_ratio = 3.0
    
    cfg.DRIFT_X = 150
    cfg.DRIFT_Y = 200
    cfg.DRIFT_PROB = 0.5
    
    return cfg
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
