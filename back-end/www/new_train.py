from i3d_learner import I3dLearner
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

model_params = {'use_cuda':True, # use cuda or not
            'use_tsm':False, # use the Temporal Shift module or not
            'use_nl':False, # use the Non-local module or not
            'use_tc':False, # use the Timeception module or not
            'use_lstm':False, # use LSTM module or not
            'freeze_i3d':False, # freeze i3d layers when training Timeception
            'batch_size_train':8, # size for each batch for training
            'batch_size_test':8, # size for each batch for testing
            'batch_size_extract_features':40, # size for each batch for extracting features
            'max_steps':100, # total number of steps for training
            'num_steps_per_update': 12, # gradient accumulation (for large batch size that does not fit into memory)
            'init_lr':1e-4, # initial learning rate
            'weight_decay':0.000001, # L2 regularization
            'momentum':0.9, # SGD parameters
            'milestones':[500, 1500], # MultiStepLR parameters
            'gamma':0.1, # MultiStepLR parameters
            'num_of_action_classes':2, # currently we only have two classes (0 and 1, which means no and yes)
            'num_steps_per_check':50, # the number of steps to save a model and log information
            'parallel':True, # use nn.DistributedDataParallel or not
            'augment':False, # use data augmentation or not
            'num_workers':1, # number of workers for the dataloader
            'mode':"rgb", # can be "rgb" or "flow" or "rgbd"
            'p_frame':"/home/paperspace/git/deep-smoke-machine/back-end/data/extracted-data/npy/rgb", # path to load video frames
            'code_testing':False # a special flag for testing if the code works
                }

fit_params = {'p_model':'/home/paperspace/git/deep-smoke-machine/back-end/data/pretrained_models/RGB-I3D-S3.pt', # the path to load the pretrained or previously self-trained model
            'model_id_suffix':"baseline-0", # the suffix appended after the model id
            'p_metadata_train':"/home/paperspace/git/deep-smoke-machine/back-end/data/split/metadata_train_split_by_date.json", # metadata path (train)
            'p_metadata_validation':"/home/paperspace/git/deep-smoke-machine/back-end/data/split/metadata_validation_split_by_date.json", # metadata path (validation)
            'p_metadata_test':"/home/paperspace/git/deep-smoke-machine/back-end/data/split/metadata_test_split_by_date.json", # metadata path (test)
            'save_model_path':"/home/paperspace/git/deep-smoke-machine/back-end/data/fine-tuned-models", # path to save the models ([model_id] will be replaed)
            'save_tensorboard_path':"/home/paperspace/git/deep-smoke-machine/back-end/data/fine-tuned-models/[model_id]/tensorboard/", # path to save data ([model_id] will be replaced)
            'save_log_path':"/home/paperspace/git/deep-smoke-machine/back-end/data/fine-tuned-models/[model_id]/train-logs/train.log", # path to save log files ([model_id] will be replaced)
            'save_metadata_path':"/home/paperspace/git/deep-smoke-machine/back-end/data/fine-tuned-models/[model_id]/metadata/" # path to save metadata ([model_id] will be replaced)
}

# #df = pd.read_json('/home/paperspace/git/deep-smoke-machine/back-end/data/metadata-fresh-air-aug-17.json')
# df = pd.read_json('/home/paperspace/git/deep-smoke-machine/back-end/data/metadata.json')
# pos = [47, 23, 19, 15]
# neg = [32, 20, 16, 12]

# dfTrain, dfTest = train_test_split(df, test_size=0.33)
# dfTrain, dfValidation = train_test_split(dfTrain, test_size=0.33)
# dfTrain.reset_index(drop=True).to_json('/home/paperspace/git/deep-smoke-machine/back-end/data/metadata_train.json', 'records')
# dfTest.reset_index(drop=True).to_json('/home/paperspace/git/deep-smoke-machine/back-end/data/metadata_test.json', 'records')
# dfValidation.reset_index(drop=True).to_json('/home/paperspace/git/deep-smoke-machine/back-end/data/metadata_val.json', 'records')



learner = I3dLearner(**model_params)
learner.fit(**fit_params)
