import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

CLASS_NAMES = (
    "Impervious surface","Clutter","Car","Tree","Low vegetation","Building"
    
    
    
    
)

color = [{"color": [255, 255, 255],},
    {"color": [0, 0, 255], },
    {"color": [0, 255, 255],},
    {"color": [0, 255, 0], },
    {"color": [255, 255, 0], },
    {"color": [255, 0, 0], },
  ]

def _get_vaihingen_meta(cat_list):
    colorlist = [i['color'] for i in color]
    ret = {
        "stuff_classes": cat_list,
        "stuff_colors": colorlist,
    }
    return ret


def register_all_vaihingen(root):
   
    meta = _get_vaihingen_meta(CLASS_NAMES)

    for name, image_dirname, sem_seg_dirname in [
        ("train", "img_dir/train", "ann_dir/train"),
        ("val", "img_dir/val", "ann_dir/val"),
    ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        all_name = f"vaihingen_sem_seg_{name}"
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


_root = os.getenv("DETECTRON2_DATASETS", "datasets/vaihingen")
register_all_vaihingen(_root)
