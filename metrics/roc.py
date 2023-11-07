import torch
from torchmetrics import ROC
from torchmetrics.functional import auc
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_roc(test_data, model):
    roc = ROC(pos_label=1)
    roc.reset()

    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for _, inputs, _, label in tqdm(test_data):
            inputs = inputs.to(device)
            label = label.to(device)
            pred = model(inputs)
            preds.append(torch.nn.functional.softmax(pred)[:,1])
            labels.append(label)
    preds = torch.cat(preds)
    labels = torch.cat(labels)

    print(f' Data num: total {len(labels)}, pos {sum(labels).item()}, neg {len(labels)-sum(labels).item()}')
    fpr, tpr, thresholds = roc(preds, labels)
    fnr = 1 - tpr
    hter_threshold = 0.22
    idx_hter = torch.argmin(abs(thresholds-hter_threshold), dim=0)
    hter = ((fpr[idx_hter] + fnr[idx_hter]) / 2).item()
    idx_eer = torch.argmin((fpr - fnr).abs(), dim=0)
    eer = fpr[idx_eer].item()
    print('HTER: {:.2%}'.format(hter))
    print('EER: {:.2%}'.format(eer))
    print('AUC: {:.2%}'.format(auc(fpr, tpr)))
    APCER = [0.1, 0.01]
    for apcer in APCER:
        bpcer = fnr[fpr>=apcer][0]
        print('BPCER@APCER={}: {:.2%}'.format(apcer, bpcer))
