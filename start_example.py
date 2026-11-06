import torch
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))
from yolo_1d.nets import training
from yolo_1d import Trainer, TrainingSettings

def main():
    
    config = {
        "project": "test_rabotaet",
        "results_path": "runs",
        "dataset": "D:/dataset/data_1d/dataset/dataset.yaml",
        # "dataset": "data/ecr_v1.yaml",
        "batch_size": 10,
        "epochs": 20,
        "armup_epochs": 2,
        "resume": False,
        "model_size": "large",
        "enable_amp": True,
        "early_stopping_epochs": 20,
        "names":["flashbutt", "alumin", "rail end", "axlec", "crossing", "manual arc"],
        # "starting_checkpoint": r"E:\git\neuro\YOLO_1d\runs\large_data4\best.pt",
        "weight_box": 10,
        "weight_dfl": 3,
        "num_workers": 4,
        "augmentations":{
            # "pipeline": "eddy",
            "regular_length": 50000, 
            "input_length": 49152, 
            "validation_resize": True,
            "random_resize": 0.1, 
            "random_crop": True,
            "random_reverse": 0, 
            "random_invert": 0.5, 
            "channel_scaling": 0.2,
            "time_masking": 0.2,
            "channel_masking": 0.2,
            "phase_spectrum_noise": 0.5,
            "random_jitter": 0.6,
            "mosaic": 0.5,
            "mix_up": 0.5,
            "sine_noise": 0.5,
            "square_noises": 0.1,
            "augment_apply_p": 0.5,
            "adversarial": 0,
        }
    }

    training.set_seed()

    t = TrainingSettings(**config)
    
    trainer = Trainer(t)
    
    trainer.train()
    trainer.validate()
    print("Testing:")
    trainer.test()

    torch.cuda.empty_cache()

def test():
    config = {
        "dataset": "D:/dataset/data_1d/dataset/dataset.yaml",
        # "dataset": "data/ecr_v1.yaml",
        "batch_size": 10,
        # "resume": True,
        "model_size": "large",
        "enable_amp": True,
        "names": ["flashbutt", "alumin", "rail end", "axlec", "crossing", "manual arc"],
        # "starting_checkpoint": r"E:\git\neuro\YOLO_1d\runs\v0.4_large1\best.pt",
        "starting_checkpoint": r"E:\git\neuro\yolo_weights\ECRv1_weights\best_new.pt"
    }

    training.set_seed()

    t = TrainingSettings(**config)
    
    trainer = Trainer(t)
    
    # trainer.train()
    # trainer.validate()
    print("Testing:")
    trainer.test()

    torch.cuda.empty_cache()

def export():
    config = {
        "dataset": "D:/dataset/data_1d/dataset/dataset.yaml",
        "batch_size": 50,
        "model_size": "small",
        "enable_amp": True,
        "names":["flashbutt", "alumin", "rail end", "axlec", "crossing", "manual arc"],
        "starting_checkpoint": r"E:\git\neuro\YOLO_1d\runs\training_08_121\best.pt",
    }

    t = TrainingSettings(**config)
    t.names
    
    trainer = Trainer(t)
    # trainer.validate()
    # trainer.test()
    print(trainer._model.head.stride)
    
    # trainer.export_onnx("YOLO_1d.onnx")

if __name__ == "__main__":
    # main()
    test()
    # export()