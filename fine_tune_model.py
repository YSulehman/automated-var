import argparse
from yolov5 import train, val, detect, export

class FineTuneModel:
    def __init__(self, data: str, batch_size: int, img_size: int, project: str = None, name: str = None):
        self.data = data
        self.batch_size = batch_size
        self.img_size = img_size
        # these two arguments are for if you want to control save location. Default is runs/train/exp/weights/best.pt
        self.project = project
        self.name = name

    def train_model(self):
        if self.project is None and self.name is None:
            train.run(imgsz=self.img_size, weights='yolov5s.pt', batch=self.batch_size, data=self.data)
        else:
            train.run(imgsz=self.img_size, weights='yolov5s.pt', batch=self.batch_size, data=self.data, project=self.project, name=self.name)

    def validate_model(self):
        if self.project is None and self.name is None:
            val.run(imgsz=self.img_size, weights='runs/train/exp/weights/best.pt', batch=self.batch_size, data=self.data)
        else:
            val.run(imgsz=self.img_size, weights=f'{self.project}/{self.name}/weights/best.pt', batch=self.batch_size, data=self.data, project=self.project, name=self.name)
        

if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=str, default='data.yaml', required=True, help="path to data.yaml file")
    parser.add_argument("--batch-size", type=int, default=16, help="batch size for training")
    parser.add_argument("--img-size", type=int, default=640, help="image size for training")
    parser.add_argument("--project", type=str, help="project name")
    parser.add_argument("--name", type=str, help="experiment name")
