from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import numpy as np

import time
import os
from six.moves import cPickle

import captioning.utils.opts as opts
import captioning.models as models
from captioning.data.dataloader import *
from captioning.data.dataloaderraw import *
import captioning.utils.eval_utils as eval_utils
import argparse
import captioning.utils.misc as utils
import captioning.modules.losses as losses
import torch


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def evaluation(model='',cnn_model='resnet101',image_folder="", infos_path='', only_lang_eval=0, force=0,device="cpu"):
    parser = argparse.ArgumentParser()      

    function_map = {"model":model, "cnn_model":cnn_model, "infos_path":infos_path, "only_lang_eval":only_lang_eval, "force":force, "device":device, "image_folder":image_folder}
    function_map = opts.add_eval_options(function_map, parser)
    function_map= opts.add_diversity_opts(function_map, parser)
    opt = Namespace(**function_map)

    # Load infos
    with open(opt.infos_path, 'rb') as f:
        infos = utils.pickle_load(f)
    # override and collect parameters
    replace = ['input_fc_dir', 'input_att_dir', 'input_box_dir', 'input_label_h5', 'input_json', 'batch_size', 'id']
    ignore = ['start_from']

    for k in vars(infos['opt']).keys():
        if k in replace:
            setattr(opt, k, getattr(opt, k) or getattr(infos['opt'], k, ''))
        elif k not in ignore:
            if not k in vars(opt):
                vars(opt).update({k: vars(infos['opt'])[k]}) # copy over options from model

    vocab = infos['vocab'] # ix -> word mapping

    pred_fn = os.path.join('eval_results/', '.saved_pred_'+ opt.id + '_' + opt.split + '.pth')
    result_fn = os.path.join('eval_results/', opt.id + '_' + opt.split + '.json')

    if only_lang_eval == 1 or (not opt.force and os.path.isfile(pred_fn)): 
        # if results existed, then skip, unless force is on
        if not opt.force:
            try:
                if os.path.isfile(result_fn):
                    json.load(open(result_fn, 'r'))
                    print('already evaluated')
                    os._exit(0)
            except:
                pass

        predictions, n_predictions = torch.load(pred_fn)
        lang_stats = eval_utils.language_eval(opt.input_json, predictions, n_predictions, vars(opt), opt.split)
        os._exit(0)

    # At this point only_lang_eval if 0
    if not opt.force:
        # Check out if 
        try:
            # if no pred exists, then continue
            tmp = torch.load(pred_fn)
            # if language_eval == 1, and no pred exists, then continue
            if opt.language_eval == 1:
                json.load(open(result_fn, 'r'))
            print('Result is already there')
            os._exit(0)
        except:
            pass

    # Setup the model
    opt.vocab = vocab
    model = models.setup(opt)
    del opt.vocab
    model.load_state_dict(torch.load(opt.model, map_location='cpu'))
    model.to(opt.device)
    model.eval()
    crit = losses.LanguageModelCriterion()
    # Create the Data Loader instance
    if len(opt.image_folder) == 0:
        loader = DataLoader(opt)
    else:
        loader = DataLoaderRaw({'folder_path': opt.image_folder, 
                            'coco_json': opt.coco_json,
                            'batch_size': opt.batch_size,
                            'cnn_model': opt.cnn_model,
                            "device": opt.device})
    # When eval using provided pretrained model, the vocab may be different from what you have in your cocotalk.json
    # So make sure to use the vocab in infos file.
    loader.dataset.ix_to_word = infos['vocab']


    # Set sample options
    opt.dataset = opt.input_json
    loss, split_predictions, lang_stats = eval_utils.eval_split(model, crit, loader, 
        vars(opt))

    return split_predictions

        

