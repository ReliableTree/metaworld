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
            'nhead':4,
            'd_hid':400,
            'd_model' : 400,
            'nlayers':4,
            'dilation' : 2,
            'output_seq':True
        },
    },
    'tailor_decoder':{
        'd_output' : 1,
        'output_seq':False
    },
    'quick_val':False,
    'val_every' : 1
}