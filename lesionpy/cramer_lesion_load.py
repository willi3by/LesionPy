import ants
import numpy as np
from natsort import natsorted
from glob import glob

def extract_lesion_load_cramer(base_path, tract_subsection_prefix, path_to_mni_lesion):
    glob_path = base_path + tract_subsection_prefix + '*'
    all_subsections = natsorted(glob(glob_path))
    sample_subsection = ants.image_read(all_subsections[0])
    lesion = ants.image_read(path_to_mni_lesion)
    lesion_resamp = ants.resample_image_to_target(lesion, sample_subsection)
    all_subsection_perc = []
    for f in all_subsections:
        subsection = ants.image_read(f)
        overlap = lesion_resamp * subsection
        perc_damage = (overlap.sum() / subsection.sum()) * 100
        all_subsection_perc.append(perc_damage)
    total_subsections_injured = np.sum(np.array(all_subsection_perc) > 5)
    perc_subsections_injured = (total_subsections_injured/16)*100
    return perc_subsections_injured