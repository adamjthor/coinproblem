from datetime import datetime
from sagemaker.mxnet.model import MXNetModel
import cv2
import os
import numpy as np
import json

#### IMPORTANT NOTE: After training the model & before running this test code, do the following:
# Download the model.tar.gz from S3
# Unzip, rename the files to: 'model-0000.params', 'model-shapes.json', 'model-symbol.json'
# - (As per this github issue: https://github.com/awslabs/amazon-sagemaker-examples/issues/216)
# Use the command line to re-zip the files as model.tar.gz
# - (Here's a helper site: http://osxdaily.com/2012/04/05/create-tar-gzip/)
# Delete the old file from S3 and upload this new one

# Set the location of the trained model & the permissions role associated with my Sagemaker notebook
my_model_data = 's3://path/to/your/model.tar.gz'
my_role = 'arn:aws:iam::############:role/service-role/AmazonSageMaker-ExecutionRole-########T######'

# Instantiate the model
my_model = MXNetModel(model_data = my_model_data,
                      role = my_role,
                      entry_point = 'entry_point.py',
                      py_version = 'py3')
print(datetime.now().time(), 'Model instantiated')

# Deploy it (based on documentation page code)
my_predictor = my_model.deploy(initial_instance_count = 1,
                               instance_type = 'ml.m4.xlarge')
print(datetime.now().time(), 'Model deployed')

# Set the working directory
os.chdir('/Users/your_username/Desktop/coinproblem')
test_set_path = 'test_set_folder_name'
test_set_folders = os.listdir(test_set_path)
test_set_folders.remove('.DS_Store')

# Store the probability of each classification {label: [img_name, label_id, pred_id, probability]}
confidence_dict = {}
# Calculate confusion matrix manually (to lighten load on memory)
cmat = np.zeros([37, 37])


# For every image label in the larger folder...
for label_id, label in enumerate(test_set_folders):

    print(datetime.now().time(), 'Working on', label, label_id)

    # ... get a list of image filenames ...
    files = os.listdir(test_set_path+'/'+label)

    # ... and for each of the image files ...
    for file in files:
        
        # Read in and transform the image
        img = cv2.imread(test_set_path+'/'+label+'/'+file)
        img = np.transpose(img, (2, 0, 1)) # change to (3, 227, 227)
        img = np.array([img]) # change to (1, 3, 227 227)
        
        # Make a prediction
        prediction = np.array(my_predictor.predict(img.tolist())).ravel()
        
        # Get index of max probability and the probability value itself
        pred_id = np.argmax(prediction)
        probability = prediction[pred_id]
        
        # Add all the info to our dictionary of classification probability info
        # pred_id (np.int64) & probability (np.float64) must be converted to Python-native types 
        dict_item = [file, label_id, pred_id.item(), probability.item()]
        if label not in confidence_dict.keys():
            confidence_dict[label] = [dict_item]
        else:
            confidence_dict[label].append(dict_item)
        
        # Increase our confusion matrix
        cmat[label_id, pred_id] += 1
        
    # Save cmat as a csv
    np.savetxt('cmat_filename.csv', cmat, delimiter=',')
    print('Confusion matrix saved')
    # Save confidence_dict as a json
    with open('confidence_dict_filename.json', 'w') as fp:
        json.dump(confidence_dict, fp)
    print('Confidence dict saved')
            

# Tear down the endpoint container and delete the corresponding endpoint configuration
my_predictor.delete_endpoint()
print(datetime.now().time(), 'Endpoint deleted')

# Delete the model
my_predictor.delete_model()
print(datetime.now().time(), 'Model deleted')