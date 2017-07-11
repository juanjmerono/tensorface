# TensorFace
TensorFlow face recognition experiment. You can train your model using tensorflow docker images with python and use the trained model with java later.

## How to use the trained model:

- Create "tests" folder to test the model.
- Add "good" and "bad" folders with jpg images inside "tests" folder.
- Run: ```mvn test```

## How to train your own model:

- Create "images" folder to train the model.
- Add one folder inside "images" for each category you want to classify with jpg images on it.
- Run docker image:
```docker run -it -p 8888:8888 -p 6006:6006 -v ${PWD}:/tf_files --workdir /tf_files floydhub/dl-docker:cpu bash```
- Now you can train the model typing:
```python retrain.py --bottleneck_dir=bottlenecks  --model_dir=inception   --summaries_dir=training_summaries/long   --output_graph=model/retrained_graph.pb   --output_labels=model/retrained_labels.txt   --image_dir=images```
- After training the model you can test it with:
```python label_image.py test.jpg```
