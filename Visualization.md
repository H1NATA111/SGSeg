## Visualization
The visualization integration of the segmentation results of the detectron2 framework is quite complex.
This method is applicable to all detectron2 frameworks.

-Find the function SemSegEvaluator under detectron2/evaluation/sem_seg_evaluation.py.
-Find 
```python
output = output["sem_seg"].argmax(dim=0).to(self._cpu_device)
```

-Insert code
```python
image_path = input["file_name"]
image_input = read_image(image_path, format="BGR")#
dataset_name = image_path.split('/')[1]
visualizer = Visualizer(image_input, MetadataCatalog.get("Vaihingen_all_sem_seg"),scale=1)#Modify based on visual datasets
vis_output = visualizer.draw_sem_seg(output, alpha=0.6)#

name = os.path.basename(image_path)
vis_output.save("pred/"+ name)
```