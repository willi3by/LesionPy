import ants
import numpy as np

def calculate_prob_weighted_lesion_load(path_to_tract, path_to_mni_lesion, return_max = False):
    tract = ants.image_read(path_to_tract)
    tract_data = tract.numpy()
    lesion = ants.image_read(path_to_mni_lesion)
    lesion_resamp = ants.resample_image_to_target(lesion, tract)
    lesion_resamp_data = lesion_resamp.numpy()
    overlap = tract_data * lesion_resamp_data
    slice_weights = [np.count_nonzero(tract_data[...,i]) for i in range(tract_data.shape[-1])]
    max_area = np.max(slice_weights)
    lesion_load = []
    for i in range(overlap.shape[-1]):
        s = np.sum(overlap[...,i])
        if slice_weights[i] == 0:
            weighted_s = 0
        else:
            weighted_s = s * (max_area / slice_weights[i])
        lesion_load.append(weighted_s)
    if return_max:
        return np.max(lesion_load), lesion_load
    else:
        lesion_load_auc = np.trapz(lesion_load)
        return lesion_load_auc, lesion_load