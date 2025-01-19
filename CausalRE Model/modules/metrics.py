import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


class Metrics:
    def __init__(self, config):
        self.config = config


    def run_metrics(self, preds, labels):
        '''
        this gets the current overall metrics for use during training

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
            precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average=self.config.f1_ave, zero_division=0)

        metrics = dict(support   = len(labels),
                       accuracy  = accuracy,
                       precision = precision,
                       recall    = recall,
                       f1        = f1)
        metrics['msg'] = f"S: {metrics['support']}\tAcc: {metrics['accuracy']:.2%}\tP: {metrics['precision']:.2%}\tR: {metrics['recall']:.2%}\tF1: {metrics['f1']:.2%}\n"        
        return metrics



    def confusion_matrix(self, preds, labels, num_classes):
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






