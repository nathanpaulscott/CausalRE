import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


def quick_metrics(preds, labels, params=None):
    '''
    gets the f1 score

    inputs:
        - aligned, cpu'd, numpy'd, flattened, filtered preds list
        - aligned, cpu'd, numpy'd, flattened, filtered labels list
        - params object

    outputs:
        - metrics dict

    '''
    if len(labels) == 0:
        accuracy, precision, recall, f1 = 0,0,0,0
    else:
        # Calculate overall  metrics
        accuracy = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true = labels, 
                                                                   y_pred = preds, 
                                                                   labels = average=params['f1_ave'], zero_division=0)

    return dict(Support   = len(labels),
                Accuracy  = accuracy,
                Precision = precision,
                Recall    = recall,
                F1        = f1)
