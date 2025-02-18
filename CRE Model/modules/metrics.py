import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix


class Metrics_skl:
    def __init__(self, config):
        self.config = config


    def run_metrics(self, flat_labels, flat_preds, placeholder=''):
        '''
        this gets the current overall metrics for use during training
        NOTE: you need to use the labels parameter to exclude the placeholders
        NOTE: it doesn't matter if labels are repeated accross batches, this will not affect the results

        inputs:
            - aligned, cpu'd, numpy'd, flattened, filtered preds list
            - aligned, cpu'd, numpy'd, flattened, filtered labels list
            - params object

        outputs:
            - metrics dict
        '''
        support = len([x for x in flat_labels if x != placeholder])
        if support == 0:
            precision, recall, f1, support = 0,0,0,0
        else:
            #labels of interest are all the unique pos labels and preds (not the placeholders)
            classes_of_interest = list(set(flat_labels + flat_preds) - {placeholder})
            #Calculate metrics
            precision, recall, f1, _ = precision_recall_fscore_support(y_true  = flat_labels, 
                                                                       y_pred  = flat_preds, 
                                                                       labels  = classes_of_interest,    #only inlcude stats for actual labels + placeholder, if you dont do this, it include stats for FP in preds, they are already accounted for by the placeholder in labels
                                                                       average = self.config.f1_ave, 
                                                                       zero_division = 0)
        metrics = dict(support   = support,
                       precision = precision,
                       recall    = recall,
                       f1        = f1)
        #add the output metrics msg
        metrics['msg'] = f"S: {metrics['support']}\tP: {metrics['precision']:.2%}\tR: {metrics['recall']:.2%}\tF1: {metrics['f1']:.2%}\n"        
        return metrics



    def confusion_matrix(self, labels, preds, num_classes):
        """ Compute confusion matrix for list inputs. """
        preds = torch.tensor(preds)
        labels = torch.tensor(labels)
        k = (labels >= 0) & (labels < num_classes)
        return torch.bincount(num_classes * labels[k] + preds[k], minlength=num_classes**2).reshape(num_classes, num_classes)



    def print_confusion_matrix(self, cm, class_labels):
        """ Print the confusion matrix with headers and labels in the command line. """
        header = "Pred\\True" + ''.join(f"{label: >6}" for label in class_labels)
        print(header)
        for i, row in enumerate(cm):
            row_text = f"{class_labels[i]}      " + ' '.join(f"{val.item():5d}" for val in row)
            print(row_text)







class Metrics_custom:
    '''
    This just does the metric calc manually fixed to 'micro' averaging.  Should be faster than the skl method
    '''
    def __init__(self, config):
        self.config = config


    def calc_metrics(self, flat_labels, flat_preds):
        labels_set = set(flat_labels)
        preds_set = set(flat_preds)
        support = len(flat_labels)
        TP = len(labels_set & preds_set)
        FN = len(labels_set - preds_set)
        FP = len(preds_set - labels_set)
        #prec
        prec = TP / (TP + FP) if (TP + FP) > 0 else 0
        #rec
        rec = TP / (TP + FN) if (TP + FN) > 0 else 0
        #f1
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

        return prec, rec, f1, support


    def run_metrics(self, flat_labels, flat_preds):
        '''
        desc

        inputs:
            - cpu'd, numpy'd, flattened, labels list
            - cpu'd, numpy'd, flattened, preds list
            - params object

        outputs:
            - metrics dict
        '''
        if len(flat_labels) == 0:
            precision, recall, f1, support = 0,0,0,0
        else:
            #Calculate metrics
            precision, recall, f1, support = self.calc_metrics(flat_labels, flat_preds)

        metrics = dict(support   = support,
                       precision = precision,
                       recall    = recall,
                       f1        = f1)
        #add the output metrics msg
        metrics['msg'] = f"S: {metrics['support']}\tP: {metrics['precision']:.2%}\tR: {metrics['recall']:.2%}\tF1: {metrics['f1']:.2%}\n"        
        return metrics

