import json
import numpy as np
from collections import defaultdict

def compare_lists(gt, pred):
    # Ensure both lists are of the same length
    if len(gt) != len(pred):
        # print("Lists must be of the same length")
        return [0]*len(gt)

    # Compare the lists and create the result list
    result = [1 if a == b else 0 for a, b in zip(gt, pred)]
    return result

def calc(results, gtfile):

    anno=json.load(open(gtfile))
    anno_dict={sample['qid']: sample['goal'] for sample in anno}

    acc=0
    holder=[]
    scores_per_class=defaultdict(list)
    scores_per_step=defaultdict(list)
    for res in results:
        pred_order=res['pred_order']
        gt_order = res['answer']
        
        # success rate acc
        if gt_order==pred_order:
            acc+=1
            scores_per_class[anno_dict[res["qid"]]].append(1)
            scores_per_step[len(res['answer'])].append(1)

        else:
            scores_per_class[anno_dict[res["qid"]]].append(0)
            scores_per_step[len(res['answer'])].append(0)

        # for step acc
        holder.extend(compare_lists(gt_order, pred_order))

    step_acc = sum(holder)/len(holder)

    scores_class={}
    for k in scores_per_class:
        scores_class[k]=round(sum(scores_per_class[k])/len(scores_per_class[k])*100, 2)

    scores_step={}
    for k in scores_per_step:
        scores_step[k]=round(sum(scores_per_step[k])/len(scores_per_step[k])*100, 2)

    final_scores = {'avg_accuracy': round(acc/len(results)*100, 2), 'avg_step_accuracy': round(step_acc*100, 2), 
                    'weighted_avg_accuracy': round(np.average([scores_class[k] for k in scores_class]), 2),
                    'scores_per_class': scores_class,
                    'scores_per_step': scores_step,
                    }

    return final_scores


