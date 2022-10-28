#!/usr/bin/env python3
from absl import app, flags
import logging
import numpy as np
import random
import os
import sys
import yaml

import torch

import utils
import experiment

utils.handle_flags()



def main(argv):    
    logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
    FLAGS = flags.FLAGS
    utils.print_flags(FLAGS)

    # Random seed initialization.
    random.seed(FLAGS.random_seed)
    np.random.seed(FLAGS.random_seed)
    torch.manual_seed(FLAGS.random_seed)
    # Configuration and paths.
    cfg = yaml.load(open('/content/QDS-Transformer/src/config.yml', 'r'), Loader=yaml.BaseLoader)
    #PATH_DATA = cfg['path_data'] 
    #PATH_CORPUS = '{}/{}'.format(PATH_DATA, cfg['corpus'])
    #PATH_DATA_PREFIX = '{}/{}'.format(PATH_DATA, cfg['data_prefix'])

    # Set up the experimental environment.
    exp = experiment.Experiment(FLAGS, cfg, dumpflag=False)

    for i, layer in enumerate(exp.model.base.encoder.layer):
        layer.attention.self.attention_window = FLAGS.window_size

    # Load the corpus.
    import json
    with open('/content/multidoc2dial/multidoc2dial_doc.json', 'r') as f:
        multidoc2dial_doc = json.load(f)
    corpus = utils.Corpus(multidoc2dial_doc, FLAGS)
    # Load train/dev data.
    #with open('/content/multidoc2dial/multidoc2dial_dial_test.json', 'r') as f:
    #    multidoc2dial_dial_test = json.load(f)
    with open('/content/multidoc2dial/multidoc2dial_dial_validation.json', 'r') as f:
        multidoc2dial_dial_validation = json.load(f)
    test_data = utils.Data(multidoc2dial_dial_validation, corpus, FLAGS)
    

    # Evaluate dev data.
    test_eval = exp.eval_dump(test_data, FLAGS.num_sample_eval,
            'Evaluating test queries')
    print('Test Evaluation', test_eval, file=sys.stderr)


if __name__ == '__main__':
    app.run(main)


