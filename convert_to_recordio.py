# Adapted from: https://mxnet.incubator.apache.org/versions/master/tutorials/basic/data.html#loading-data-using-image-iterators
# Modified slightly from 'subprocess.check_call' to 'subprocess.run'

import mxnet as mx
import os
import subprocess

# Define directory information
im2rec_path = mx.test_utils.get_im2rec_path()
data_path_train = os.path.join(os.getcwd(),'your_training_data_directory')
data_path_val = os.path.join(os.getcwd(),'your_validation_data_directory')
data_path_test = os.path.join(os.getcwd(),'your_test_data_directory')
prefix_path_train = 'your_destination_path_train'
prefix_path_val = 'your_destination_path_validation'
prefix_path_test = 'your_destination_path_test'

# Create .lst file for TEST
with open(os.devnull, 'wb') as devnull:
    subprocess.run(['python', im2rec_path, '--list', '--recursive', '--test-ratio=0', prefix_path_test, data_path_test], 
                   stdout=devnull)
print('Done LST test')  
  
# Create recordio file for TEST
with open(os.devnull, 'wb') as devnull:
    subprocess.run(['python', im2rec_path, '--num-thread=4', '--pass-through', '--test-ratio=0', prefix_path_test, data_path_test],
                   stdout=devnull)
print('Done RecordIO test')

# Create .lst file for VALIDATION
with open(os.devnull, 'wb') as devnull:
    subprocess.run(['python', im2rec_path, '--list', '--recursive', '--test-ratio=0', prefix_path_val, data_path_val], 
                   stdout=devnull)
print('Done LST validation')  
  
# Create recordio file for VALIDATION
with open(os.devnull, 'wb') as devnull:
    subprocess.run(['python', im2rec_path, '--num-thread=4', '--pass-through', '--test-ratio=0', prefix_path_val, data_path_val],
                   stdout=devnull)
print('Done RecordIO validation')

# Create .lst file for TRAIN
with open(os.devnull, 'wb') as devnull:
    subprocess.run(['python', im2rec_path, '--list', '--recursive', '--test-ratio=0', prefix_path_train, data_path_train], 
                   stdout=devnull)
print('Done LST train')    

# Create recordio file for TRAIN
with open(os.devnull, 'wb') as devnull:
    subprocess.run(['python', im2rec_path, '--num-thread=4', '--pass-through', '--test-ratio=0', prefix_path_train, data_path_train],
                   stdout=devnull)
print('Done RecordIO train')