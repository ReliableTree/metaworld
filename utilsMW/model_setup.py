model_setup = {
    'obj_embedding': {'use_obj_embedding':True, 'train_embedding':True, 'EIS':30, 'EOS':10},
    'attn_trans' : {'use_attn_trans':True},
    'train'      : True,
    'use_memory' : False,
    'meta_world' : {
        'use'   :True,
        'seq_len': 200
    },
    'lang_trans' :  {
        'use_lang_trans' : True,
        'd_output' : 30,
        'd_model'  : 42,
        'nhead'    : 2,
        'nlayers'  : 2,
        'bottleneck' : False
    },
    'contr_trans': {
        'use_contr_trans':True,
        'd_output'   : 5,
        'd_model'    : 210,
        'nhead'      : 6,
        'nlayers'    : 4,
        'recursive'    : False,
        'use_gen2'     : False,
        'use_mask'     : False,
        'use_counter_embedding': False,
        'count_emb_dim' : 20,
        'predictionNN'  : False,
        'plan_nn'       : {
            'use_plan_nn'   : True,
            'plan'     :{
                'use_layernorm':False,
                'plan_type' : 'upconv',
                'num_upconvs':5,
                'stride':3,
                'd_output':5,
                'nhead':4,
                'd_hid':40,
                'nlayers':2
            },
        }

    },
    'LSTM':{
        'use_LSTM' : False
    },
    'quick_val':False,
    'val_every' : 20
}