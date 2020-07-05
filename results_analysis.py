import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from statistics import mode

# Analyze 'confidence_dict_filename.json' to determine which label corresponds to which index
# dict_item = [[file, label_id, pred_id.item(), probability.item()], ...]
# dict key is the true label

with open('confidence_dict_filename.json', 'r') as f:
    conf_dict = json.load(f)



# Create 'results' dictionary 
# {true_label: 
#   {'label_id': ___, 
#    'mode_pred_id': ___, 
#    'probabilities': []}   [prob for prob in ]
results = {}

for key in conf_dict.keys():
    # Define result items for each true coin type
    label_id = conf_dict[key][0][1]
    pred_ids = [pred[2] for pred in conf_dict[key]]
    mode_pred_id = mode(pred_ids)
    probabilities = [pred[3] for pred in conf_dict[key]]
    # Add them to results dictionary
    results[key] = {'label_id': label_id, 'pred_ids': pred_ids, 'mode_pred_id': mode_pred_id, 'probs': probabilities}

with open('predictions_per_label.json', 'w') as f: 
    json.dump(results, f)
    
    
#### Make histograms of prediction confidences for each coin class and save to file.
# Note that if we fix number of bins AND x-range, then any histogram where the range is too small 
# between max & min values will have bars whose widths are too small too see.
# To get around this, we calculate the span of prediction confidences for each coin class,
# and set the x-tick properties based on this span.
# This has the side effect though of histograms having wildly different x-ranges. 
for key in conf_dict.keys():
    # Create & save normal-axis histogram
    plt.figure()
    plt.hist(results[key]['probs'], bins=100)
    plt.title(key+' prediction confidences')
    plt.ylabel('Count of images')
    span = round(np.array(results[key]['probs']).max() - np.array(results[key]['probs']).min(), 5)
    plt.xlabel('Prediction confidence, range=' + str(span))
    plt.locator_params(axis='x', nbins=5)
    plt.ticklabel_format(axis='x', useOffset=False)
    plt.ylim(0,1621)
    plt.savefig('HT_probability_histograms/'+key+'.png', dpi=192)
       
    # Create & save y-log-axis histogram
    plt.figure()
    plt.hist(results[key]['probs'], bins=100)
    plt.yscale('log')
    plt.title(key+' prediction confidences (log)')
    plt.ylabel('Count of images (log scale)')
    span = round(np.array(results[key]['probs']).max() - np.array(results[key]['probs']).min(), 5)
    plt.xlabel('Prediction confidence, range=' + str(span))
    plt.locator_params(axis='x', nbins=5)
    plt.ticklabel_format(axis='x', useOffset=False)
    plt.ylim(0.1621)
    plt.savefig('HT_probability_histograms/'+key+'_log.png', dpi=192)




# Make confusion matrix, now that I know which label_id corresponds to which coin_label
df = pd.read_csv('cmat_filename.csv', header=0, index_col=0)
# Rows are actual label_id, columns are predicted_id
# Define proper column order & redefine the dataframe to match
proper_col_order = [str(results[key]['mode_pred_id']) for key in results.keys() if key != 'Others']
df = df[proper_col_order]
# Label the numerical row & column indices with the coin labels
df.index = results.keys()
df.columns = [key for key in results.keys() if key != 'Others']
# Save to file
df.to_csv('HT_cmat.csv')



# Merge H/T results based on coin type (only the probabilities matter here)
results_merged = {}

for key in results.keys():
    if key == 'Others':
        results_merged[key] = results[key]
    else:
        if key[:-1] not in results_merged.keys():
            results_merged[key[:-1]] = results[key]
        else:
            # Create a probability list that includes Heads & Tails predictions
            results_merged[key[:-1]]['probs'] += results[key]['probs']

# Create & save histograms
for key in results_merged.keys():
    # Create & save normal-axis histogram
    plt.figure()
    plt.hist(results_merged[key]['probs'], bins=100)
    plt.title(key+' prediction confidences')
    plt.ylabel('Count of images')
    span = round(np.array(results_merged[key]['probs']).max() - np.array(results_merged[key]['probs']).min(), 5)
    plt.xlabel('Prediction confidence, range=' + str(span))
    plt.locator_params(axis='x', nbins=5)
    plt.ticklabel_format(axis='x', useOffset=False)
    plt.ylim(0,3241)
    plt.savefig('merged_probability_histograms/'+key+'.png', dpi=192)
    
    # Create & save y-log-axis histogram
    plt.figure()
    plt.hist(results_merged[key]['probs'], bins=100)
    plt.yscale('log')
    plt.title(key+' prediction confidences (log)')
    plt.ylabel('Count of images (log scale)')
    span = round(np.array(results_merged[key]['probs']).max() - np.array(results_merged[key]['probs']).min(), 5)
    plt.xlabel('Prediction confidence, range=' + str(span))
    plt.locator_params(axis='x', nbins=5)
    plt.ticklabel_format(axis='x', useOffset=False)
    plt.ylim(0.3241)
    plt.savefig('merged_probability_histograms/'+key+'_log.png', dpi=192)





#### Create & save confusion matrices

# Calculate merged columns
df['1S4'] = df['1S4H'] + df['1S4T']
df['10S2'] = df['10S2H'] + df['10S2T']
df['10S1'] = df['10S1H'] + df['10S1T']
df['1S3'] = df['1S3H'] + df['1S3T']
df['2S1'] = df['2S1H'] + df['2S1T']
df['50pS1'] = df['50pS1H'] + df['50pS1T']
df['2S2'] = df['2S2H'] + df['2S2T']
df['50pS2'] = df['50pS2H'] + df['50pS2T']
df['5S3'] = df['5S3H'] + df['5S3T']
df['2S3'] = df['2S3H'] + df['2S3T']
df['5S2'] = df['5S2H'] + df['5S2T']
df['1S1'] = df['1S1H'] + df['1S1T']
df['5S1'] = df['5S1H'] + df['5S1T']
df['1S2'] = df['1S2H'] + df['1S2T']
df['2S4'] = df['2S4H'] + df['2S4T']
df['1S6'] = df['1S6H'] + df['1S6T']
df['1S5'] = df['1S5H'] + df['1S5T']
df['10S3'] = df['10S3H'] + df['10S3T']

# Define columns to keep and reset dataframe
coin_classes = ['1S4', '10S2', '10S1', '1S3', '2S1', '50pS1', '2S2', 'Others', '50pS2', '5S3', '2S3', '5S2', '1S1', '5S1', '1S2', '2S4', '1S6', '1S5', '10S3']
coin_columns = [coin for coin in coin_classes if coin != 'Others']
df = df[coin_columns]

# Add rows of corresponding classes together
new_rows = []

for coin in coin_classes:
    if coin == 'Others':
        new_rows.append(list(df.loc[coin,:]))
    else:
        new_rows.append(list(df.loc[coin+'H',:] + df.loc[coin+'T',:]))
        
# Create & save dataframe (after moving Others to the bottom)
df_new = pd.DataFrame(new_rows, columns=coin_columns, index=coin_classes, dtype=int)
a = df_new.loc[[i for i in df_new.index if i != 'Others'], :]
b = df_new.loc['Others', :]
df_new = pd.concat([a, b], axis=1).T
df_new.to_csv('merged_cmat.csv')




# Create & save % accuracy version of confusion matrix
df_pct = df_new.copy()

# For the classes which have true&predicted, sum the columns
for i in range(len(df_pct.columns)):
    df_pct.iloc[:-1, i] = df_pct.iloc[:-1, i]/3240

# For Others, sum the row, just to get a sense of how its mislabellings are distributed
df_pct.loc['Others',:] = df_pct.loc['Others', :]/sum(df_new.loc['Others',:])

df_pct.to_csv('merged_pct_cmat.csv')





#### Create full CSV of every test set prediction:
# Filename | True label | Predicted label | Probability

# Load in the fixed conf dict
# {'label': [[file1, labelid1, predid1, prob1], [2], [3], [4], ...]}
with open('confidence_dict_filename.json', 'r') as f:
    conf_dict = json.load(f)
    
# Create dictionary mapping pred_label_ids to label_names
mapping_dict = {}

for key in conf_dict.keys():
    if key != 'Others':
        # Define result items for each true coin type
        pred_ids = [pred[2] for pred in conf_dict[key]]
        mode_pred_id = mode(pred_ids)
        
        mapping_dict[mode_pred_id] = key

# Now create a 'pred_label' entry by matching the 'pred_id' of a prediction with the 'true_label'
pred_lists = []

for true_label in conf_dict.keys():
    for i in range(len(conf_dict[true_label])):
        filename = conf_dict[true_label][i][0]
        pred_id = conf_dict[true_label][i][2]
        pred_label = mapping_dict[pred_id]
        probability = conf_dict[true_label][i][3]
        
        pred_lists.append([filename, true_label, pred_label, probability])
        
        
# Create dataframe and save it
df_preds = pd.DataFrame(pred_lists, columns=['filename','true_label','pred_label','probability'])
df_preds.to_csv('test_set_predictions_w_probabilities.csv')


