model_setup = {
    'obj_embedding': {'use_obj_embedding':True, 'train_embedding':True, 'EIS':30, 'EOS':10},
    'attn_trans' : {'use_attn_trans':True},
    'train'      : True,
    'use_memory' : False,
    'meta_world' : {
        'use'   :True,
    },
    'lang_trans' :  {
        'use_lang_trans' : True,
        'd_output' : 30,
        'd_model'  : 42,
        'nhead'    : 2,
        'nlayers'  : 2,
        'bottleneck' : False
    },
    'plan_nn'       : {
        'use_plan_nn'   : True,
        'plan'     :{
            'use_layernorm':False,
            'upconv' : True,
            'num_upconvs':5,
            'stride':3,
            'd_output':4,
            'nhead':10,
            'd_hid':200,
            'd_model' : 200,
            'nlayers':6,
            'seq_len': 200,
            'dilation' : 2
        },
    },
    'tailor_transformer': {
                'use_layernorm':False,
                'upconv' : False,
                'num_upconvs':4,
                'stride':3,
                'd_output':5,
                'd_model' : 60,
                'nhead':6,
                'd_hid':60,
                'nlayers':4,
                'd_result':2,
                'seq_len': 200,

    },
    'quick_val':False,
    'val_every' : 10000,
    'seq_len': 200
}