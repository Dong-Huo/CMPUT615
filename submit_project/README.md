Author Yuchen Yang, Dong Huo

Acknowledge:
Work distribution can be divided into: 

(1) Yuchen Yang: Pipelines design, Model and loss strategies implementation, train/test/eval interfaces implementation. Main report & presentation writing. 

./model_train_test_eval
data/------------------------put generated data folder(for train) and original VOT14,VOT16(for test) folder here.
b64_0.00001_alex_model/------contain the model we trained and used for presentation (100.model)
mode/------------------------train models storage
AlexNet_model.py-------------Alexnet model implementation using pytroch
Config.py--------------------main configurations (train parameters, path, warping style)
read_data.py-----------------train/test data reading interface for pytorch model
siamese_module.py------------the siamese model of our framework. 3 strategies implementation using pytorch.
test_eval.py-----------------test code for testing in the training data (search window cropped in advanced)
test_eval_full.py------------test code for testing on real video (search window cropped online)
train.py---------------------code for training the model. 

 
(2) Dong Huo: Training set and label generation, Overview of wrapping pipeline implementation, Evaluation criterion implementation.

./data_generation:
dlt.py-----------------------DLT calculations using opencv
generate_data.py-------------generate data without augmentation
generate_scale.py------------generate augmented data with random scale
generate_trans.py------------generate augmented data with random translation
generate_both.py-------------generate augmented data with random scale and translation 

./model_train_test_eval
data_stuff.py----------------overview warpping and (EAO) evaluation code (relative coords <-> absolute coords)
dlt.py-----------------------DLT calculations using opencv


Requirements:

Ubuntu 14.04 +
python3.6
opencv
pytorch (0.4.0 with cuda)
sharply
torchvision
numpy


Codes used for reference:
https://github.com/chenhsuanlin/inverse-compositional-STN
https://github.com/harveyslash/Facial-Similarity-with-Siamese-Networks-in-Pytorch



model link:
https://drive.google.com/file/d/19sDU6Mgpax9S_5gHFG6Ft_GPiXxPxudL/view?usp=sharing

