import torch
import numpy as np

def geodesic_uncertainty(model, samples):
    preds = []
    class_count = None
    for sample in samples:
        sample = sample.reshape([1, 1, 28, 28])
        pred = torch.softmax(model(sample), dim=1)[0] 
        preds.append(torch.argmax(model(sample)).item())
        if class_count is None:
            class_count = pred.detach().numpy()
        else:
            class_count += pred.detach().numpy()

    print(class_count)

    entropy = 0
    num_classes = len(class_count)
    num_pred = len(preds)
    for i in range(num_classes):
        p_class = class_count[i] / num_pred
        entropy -= p_class * np.log(p_class)
    
    var = np.var(preds)

    max_entropy = np.log(num_classes)
    entropy_uncertainty = entropy / max_entropy

    return preds, var, entropy, entropy_uncertainty