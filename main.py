

import pickle
from deltaencoder import DeltaEncoder


########### Load Data ################
features_train, labels_train, features_test, labels_test, episodes_1shot, episodes_5shot = pickle.load(open('data/mIN.pkl','rb'))

# features_train/features_test are features extracted from some backbone (resnet18); they are np array with size = (N,D), where N is the number of samples and D the features dimensions
# labels_train/labels_test are one hot GT labels with size = (N,C), where C is the number of classes (can be different for train and test sets
# episodes_*shot are supplied for reproduction of the paper results size=(num_episodes, num_classes, num_shots, D)


# import numpy as np
# features_test_set = set(["".join(x.astype('str')) for x in features_test])
# features_train_set = set(["".join(x.astype('str')) for x in features_train])
# for episode in episodes_1shot:
#         features_1shot = np.reshape(episode[0], (episode[0].shape[0]*episode[0].shape[1], episode[0].shape[2]))
#         features_1shot_set = set(["".join(x.astype('str')) for x in features_1shot])
#         print("episode 1 intersect with test: {}".format(len(features_test_set & features_1shot_set)))
#         print("episode 1 intersect with train: {}".format(len(features_train_set & features_1shot_set)))
#
# for episode in episodes_5shot:
#         features_5shot = np.reshape(episode[0], (episode[0].shape[0]*episode[0].shape[1], episode[0].shape[2]))
#         features_5shot_set = set(["".join(x.astype('str')) for x in features_5shot])
#         print("episode 5 intersect with test: {}".format(len(features_test_set & features_5shot_set)))
#         print("episode 5 intersect with train: {}".format(len(features_train_set & features_5shot_set)))
#
# exit()

# note for args:
# nb_img is the number of images(features in fact) to be generated, to be passed later in the linear classifier
# drop_out_rate_input: only applied in the features (X and not referenence features as well) before given to the encoder
# drop_out_rate: applied to every layer of encoder / decoder
# noise_size: the size of the encoding (Z)
# nb_val_loop: unused

######### 1-shot Experiment #########
args = {'data_set' : 'mIN',
        'num_shots' : 1,
        'num_epoch': 6,
        'nb_val_loop': 10,
        'learning_rate': 1e-5, 
        'drop_out_rate': 0.5,
        'drop_out_rate_input': 0.0,
        'batch_size': 128,
        'noise_size' : 16,
        'nb_img' : 1024,
        'num_ways' : 5,
        'encoder_size' : [8192],
        'decoder_size' : [8192],
        'opt_type': 'adam'
       }

model = DeltaEncoder(args, features_train, labels_train, features_test, labels_test, episodes_1shot)
model.train(verbose=True)


######### 5-shot Experiment #########
args = {'data_set' : 'mIN',
        'num_shots' : 5,
        'num_epoch': 12,
        'nb_val_loop': 10,
        'learning_rate': 1e-5,
        'drop_out_rate': 0.5,
        'drop_out_rate_input': 0.0,
        'batch_size': 128,
        'noise_size' : 16,
        'nb_img' : 1024,
        'num_ways' : 5,
        'encoder_size' : [8192],
        'decoder_size' : [8192],
        'opt_type': 'adam'
       }

model = DeltaEncoder(args, features_train, labels_train, features_test, labels_test, episodes_5shot)
model.train(verbose=True)
