"""
SUN dataset preprocessing for CZSL
"""

import os
import scipy.io
import numpy as np

sun_ds_dir = 'data/SUN'
attrs_mat = scipy.io.loadmat(f'{sun_ds_dir}/SUNAttributeDB/attributeLabels_continuous.mat')
images_mat = scipy.io.loadmat(f'{sun_ds_dir}/SUNAttributeDB/images.mat')
# attrs = scipy.io.loadmat(f'{sun_ds_dir}/SUNAttributeDB/attributes.mat')

# Step 1. Preprocess attributes
attrs = attrs_mat['labels_cv']
images = [im[0][0] for im in images_mat['images']]
class_names = [os.path.dirname(img) for img in images]
unique_class_names = sorted(list(set(class_names)))
labels = np.array([unique_class_names.index(c) for c in class_names])
n_classes = len(unique_class_names)

class_idx = [np.where(labels == c)[0] for c in range(n_classes)]
class_attrs = np.array([attrs[idx].mean(axis=0) for idx in class_idx])
class_attrs = class_attrs / np.linalg.norm(class_attrs, axis=1, keepdims=True)
np.save('data/attributes', class_attrs)
np.save('data/image_files', images)
np.save('data/labels', labels)

# Step 2. Create the splits
# There are 20 imgs per category. Let's devote 10 imgs per class for train,
# 5 imgs per class for val and 5 imgs per class for val.
rs = np.random.RandomState(42)

shuffled_class_idx = [rs.permutation(idx) for idx in class_idx]
train_idx = [i for idx in shuffled_class_idx for i in idx[:10]]
val_idx = [i for idx in shuffled_class_idx for i in idx[10:15]]
test_idx = [i for idx in shuffled_class_idx for i in idx[15:]]
len(set(train_idx + val_idx + test_idx)) == 717 * 20
np.save('data/SUN/train_idx', train_idx)
np.save('data/SUN/val_idx', val_idx)
np.save('data/SUN/test_idx', test_idx)
