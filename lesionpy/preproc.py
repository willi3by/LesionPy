import ants
import numpy as np
import nibabel as nib
import cv2
import math
import scipy.ndimage as ndimage

def convert_analyze_to_nii(path_to_mni_cst_template, path_to_analyze_segmentation, new_fn):
    im1 = nib.load(path_to_mni_cst_template)
    im2 = nib.load(path_to_analyze_segmentation)
    new_img = im2.__class__(im2.dataobj[:], im1.affine, im1.header)
    new_img.header['pixdim'] = im2.header['pixdim']
    nib.save(new_img, new_fn)
    return new_img

def radial_split_tract(path_to_tract, N, hemisphere):
    tract = ants.image_read(path_to_tract)
    # lesion = ants.image_read(path_to_lesion)
    tract_arr = tract.numpy()
    # lesion_arr = lesion.numpy()

    h, w = tract_arr.shape[0], tract_arr.shape[1]  # image height and width
    N = N  # number of slices in our pie
    l = h + w  # length of radial lines - larger than necessary
    for sector in range(N):
        sector_per_slice = []
        for s in range(tract_arr.shape[2]):
            tract_slice = tract_arr[:, :, s]
            cx, cy = ndimage.measurements.center_of_mass(tract_slice)  # (x,y) coordinates of circle centre
            if np.isnan(cx) or np.isnan(cy):
                cx = 0
                cy = 0
            startAngle = sector * 360 / N
            endAngle = startAngle + 360 / N
            x1 = round(cx) + l * math.sin(math.radians(startAngle))
            y1 = round(cy) - l * math.cos(math.radians(startAngle))
            x2 = round(cx) + l * math.sin(math.radians(endAngle))
            y2 = round(cy) - l * math.cos(math.radians(endAngle))
            vertices = [(cy, cx), (y1, x1), (y2, x2)]
            # Make empty black canvas
            im = np.zeros((h, w), np.uint8)
            # Draw this pie slice in white
            cv2.fillPoly(im, np.array([vertices], 'int32'), 255)
            subsection = cv2.bitwise_and(im.astype(np.int32), tract_slice.astype(np.int32))
            sector_per_slice.append(subsection)
            sector_mask = np.stack(sector_per_slice, axis=2)
            sector_ants = ants.from_numpy(sector_mask.astype(np.float32))
            sector_ants = ants.copy_image_info(tract, sector_ants)
            sector_fn = hemisphere + '_sector_' + str(sector) + '.nii'
            ants.image_write(sector_ants, sector_fn)