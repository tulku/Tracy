# Tracy: the dice tracing app

To install dependencies:

```shell
$ poetry install
```

# Using YOLOv8

## Training

```shell
$ poetry shell
$ yolo train data=/home/tulku/Repos/Tracy/dataset/data.yaml model=yolov8s.pt epochs=40 lr0=0.01 batch=15
```

## Prediction

```shell
$ poetry shell
$ yolo predict show=True model=<path_to>/best.pt source=<image to predict>
```
