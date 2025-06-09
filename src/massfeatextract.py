import os 
import pandas as pd
import numpy as np
from skimage import io
from datetime import date
from tqdm import tqdm
import pyfeats

def feature_extract(f, mask):
    glcm_features_mean, glcm_features_range, glcm_labels_mean, glcm_labels_range = pyfeats.glcm_features(
        f, ignore_zeros=False
    )
    glds_features, glds_labels = pyfeats.glds_features(f, mask, Dx=[0, 1, 1, 1], Dy=[1, 1, 0, -1])
    lbp_features, lbp_labels = pyfeats.lbp_features(f, mask, P=[8, 16, 24], R=[1, 2, 3])
    zernike_features, zernike_labels = pyfeats.zernikes_moments(f, radius=9)
    hu_features, hu_labels = pyfeats.hu_moments(f)
    tas_features, tas_labels = pyfeats.tas_features(f)

    all_features = np.concatenate([
        glcm_features_mean, glcm_features_range,
        glds_features, lbp_features,
        zernike_features, hu_features, tas_features
    ])
    all_labels = (
        glcm_labels_mean + glcm_labels_range +
        glds_labels + lbp_labels +
        zernike_labels + hu_labels + tas_labels
    )
    return pd.DataFrame([all_features], columns=all_labels)


def process_texture_dataset(root_dir, dataset_name):
    df_list = []
    for class_name in tqdm(os.listdir(root_dir), desc=f"Processing {dataset_name}"):
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            if not os.path.isfile(img_path):
                continue
            try:
                img = io.imread(img_path, as_gray=True)
                if img.dtype != np.uint8:
                    img = (img * 255).clip(0, 255).astype(np.uint8)
                mask = np.ones_like(img, dtype=np.uint8)

                df = feature_extract(img, mask)
                df['Image'] = img_name
                df['Folder'] = class_name
                df['Dataset'] = dataset_name
                df_list.append(df)
            except Exception as e:
                print(f"Skipping {img_name}: {e}")
    return pd.concat(df_list, ignore_index=True)


if __name__ == "__main__":
    today = date.today().isoformat()
    dtd_path = './image_sets/dtd'
    svs_path = './image_sets/severstal'

    all_data = pd.DataFrame()

    if os.path.exists(dtd_path):
        all_data = pd.concat([all_data, process_texture_dataset(dtd_path, "dtd")])
    if os.path.exists(svs_path):
        all_data = pd.concat([all_data, process_texture_dataset(svs_path, "severstal")])

    os.makedirs("data", exist_ok=True)
    outname = os.path.join("data", f"{today}_texture_features_severstal.xlsx")
    all_data.to_excel(outname, index=False)
    print(f"Saved extracted features to {outname}")
