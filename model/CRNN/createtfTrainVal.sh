#! /bin/bash
# int-or-string.sh
#06Dual/tfReal
#python3 createFsnsTest2.py train_crnn.txt data
#mv data/train_feature.tfrecords data/tfReal/train_feature.tfrecords
#python3 createFsnsTest2.py test_crnn.txt data
#mv data/train_feature.tfrecords data/tfReal/test_feature.tfrecords
#python3 createFsnsTest2.py val_crnn.txt data
#mv data/train_feature.tfrecords data/tfReal/validation_feature.tfrecords
#python3 tools/write_text_features.py --dataset_dir data/ --save_dir data/tfReal/
./TrainVal.sh > data/tfReal/logTrain_afterepoch4 2> data/tfReal/log2Train_afterepoch4
#./TrainVal.sh
