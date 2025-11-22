import torch
import numpy as np

def coefficient_of_variation(mean, var, eps=1e-8):
    """
    This is a way to quantify relative variance or rather relative standard deviation.
    """
    return np.sqrt(var) / (np.abs(mean) + eps)

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

    entropy = 0
    num_classes = len(class_count)
    num_pred = len(preds)
    for i in range(num_classes):
        p_class = class_count[i] / num_pred
        entropy -= p_class * np.log(p_class)
    
    var = np.var(preds)

    cov = coefficient_of_variation(np.mean(preds), var)

    max_entropy = np.log(num_classes)
    entropy_uncertainty = entropy / max_entropy

    return preds, var, cov, entropy, entropy_uncertainty

def calculate_intraclass_geodesic_distances(points_dict, samples=50):
    """
    This method takes the 50 first instances from each class and checks the average geodesic distance within each class. 
    This distance could then be compared to the distance from an input and samples from a predicted class.
    """
    # TODO: collect both a lower diagonal matrix of distances for 50 samples and an array with 10 intraclass distance averages
    pass
    

def intergeodesic_uncertainty(model, samples):
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

    entropy = 0
    num_classes = len(class_count)
    num_pred = len(preds)
    for i in range(num_classes):
        p_class = class_count[i] / num_pred
        entropy -= p_class * np.log(p_class)
    
    var = np.var(preds)

    cov = coefficient_of_variation(np.mean(preds), var)

    max_entropy = np.log(num_classes)
    entropy_uncertainty = entropy / max_entropy

    return preds, var, cov, entropy, entropy_uncertainty


