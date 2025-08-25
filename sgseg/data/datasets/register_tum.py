import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

tum_CATEGORIES = [
    {"color": [0, 0, 63], "id": 1, "name": "Background"},
    {"color": [0, 63, 63], "id": 2, "name": "Arable land"},
    {"color": [0, 63, 0], "id": 3, "name": "Permanent crops"},
    {"color": [0, 63, 127], "id": 4, "name": "Pastures"},
    {"color": [0, 63, 191], "id": 5, "name": "Forests"},
    {"color": [0, 63, 255], "id": 6, "name": "Surface water"},
    {"color": [0, 127, 63], "id": 7, "name": "Shrub"},
    {"color": [0, 127, 127], "id": 8, "name": "Open spaces"},
    {"color": [0, 0, 127], "id": 9, "name": "Wetlands"},
    {"color": [0, 0, 191], "id": 10, "name": "Mine, dump"},
    {"color": [0, 0, 255], "id": 11, "name": "Artificial vegetation"},
    {"color": [0, 191, 127], "id": 12, "name": "Urban fabric"},
    {"color": [0, 127, 191], "id": 13, "name": "Buildings"},
]

def _get_tum_meta():
    stuff_ids = [k["id"] for k in tum_CATEGORIES]
    stuff_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(stuff_ids)}
    stuff_classes = [k["name"] for k in tum_CATEGORIES]
    stuff_colors = [k["color"] for k in tum_CATEGORIES]

    ret = {
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes,
        "stuff_colors": stuff_colors,
    }
    return ret

def register_tum(root):
    meta = _get_tum_meta()
    for name, image_dirname, sem_seg_dirname in [
        ("val", "TUM_128/val/rgb", "TUM_128/val/newlabel2"),
        ("train", "TUM_128/val/rgb", "TUM_128/val/newlabeltrain"),
    ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        name = f"tum_{name}_sem_seg"
        DatasetCatalog.register(
            name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="png")
        )
        MetadataCatalog.get(name).set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
            **meta,
        )

root = "/data/anke/Code/DeCLIP-DeCLIP_CATSeg/datasets"
register_tum(root)
