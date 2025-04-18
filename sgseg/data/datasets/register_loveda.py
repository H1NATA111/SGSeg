import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

CLASS_NAMES = (
    "No-data",
    "Background",
    "Building",
    "Road",
    "Water",
    "Barren",
    "Forest",
    "Agriculture" 
)

color = [{"color": [0, 0, 0],},
    {"color": [0, 0, 63], },
    {"color": [0, 63, 63],},
    {"color": [0, 63, 0], },
    {"color": [0, 63, 127], },
    {"color": [0, 63, 191], },
    {"color": [0, 63, 255], },
    {"color": [0, 127, 63], },
    ]

def _get_loveda_meta(cat_list):
    colorlist = [i['color'] for i in color]
    ret = {
        "stuff_classes": cat_list,
        "stuff_colors": colorlist,
    }
    return ret


def register_all_loveda_33k(root):
    # root = os.path.join(root, "VOC2012")
    meta = _get_loveda_meta(CLASS_NAMES)

    for name, image_dirname, sem_seg_dirname in [
        ("train", "img_dir/train", "ann_dir/train"),
        ("val", "img_dir/val", "ann_dir/val"),
    ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        all_name = f"loveda_sem_seg_{name}"
        DatasetCatalog.register(
            all_name,
            lambda x=image_dir, y=gt_dir: load_sem_seg(
                y, x, gt_ext="png", image_ext="png"
            ),
        )
        MetadataCatalog.get(all_name).set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
            **meta,
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets/loveda")
register_all_loveda_33k(_root)
