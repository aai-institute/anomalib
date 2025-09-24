import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# General imports
import numpy as np
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from matplotlib import pyplot as plt
from PIL import Image
from pathlib import Path
from glob import glob
import pandas as pd

from tqdm import tqdm
import time


# Anomalib imports
# Import AllInOneBlock
from anomalib.models.components.flow import AllInOneBlock 
from anomalib.data import MVTec, Kolektor, BTech, Visa
from anomalib.engine import Engine
from anomalib.models import (
    Cfa,
    Cflow,
    Csflow,
    Dfkde,
    Dfm,
    Draem,
    Dsr,
    EfficientAd,
    Fastflow,
    Fre,
    Ganomaly,
    Padim,
    Patchcore,
    ReverseDistillation,
    Stfpm,
    Supersimplenet,
    Uflow,
    VlmAd,
    WinClip
)
from anomalib import metrics
from anomalib.utils.post_processing import superimpose_anomaly_map
from anomalib.data.datasets.image.visa import CATEGORIES
from anomalib.data.datasets.image.mvtec import CATEGORIES as MVTec_CATEGORIES


rows = 3
cols = 3
max_epochs = 100
CATEGORIES = CATEGORIES

# Make Sure that prefix relects the experimental setup, e.g. the used flow layer
prefix = "ablation_mvtec"


results = pd.DataFrame()
for model_cls in tqdm(["UFlow", "FastFlow", "CFlow"], desc="Models"):
    for cls in tqdm(MVTec_CATEGORIES, desc="Classes", leave=False):
        
        models_types = {
            "UFlow": {
                "class": Uflow,
                "callbacks": [
                    ModelCheckpoint(
                        mode="min",
                        monitor='loss',
                    ),
                    EarlyStopping(
                        monitor='loss',
                        mode="min",
                        patience=3,
                    ),    
                ] 
            },
            "FastFlow": {
                "class": Fastflow,
                "callbacks": [
                    ModelCheckpoint(
                        mode="max",
                        monitor='pixel_AUROC',
                    ),
                    EarlyStopping(
                        monitor='pixel_AUROC',
                        mode="max",
                        patience=3,
                    ),
                ] 
            },
            "CFlow": {
                "class": Cflow,
                "callbacks": [
                    ModelCheckpoint(
                        mode="min",
                        monitor='train_loss_epoch',
                    ),
                    EarlyStopping(
                        monitor='train_loss_epoch',
                        mode="min",
                        patience=3,
                    ),
                ] 
            }
        }
        print("+"*80)
        print(f"Training {model_cls} on class {cls}")
        print("+"*80)
        dataset_root = Path.cwd() / "datasets" / "Mvtec"
        
        datamodule = MVTec(
                root = dataset_root,
                category=cls,
                train_batch_size=32,
                eval_batch_size=32,
                num_workers=8,
        )
        
        #model = Fastflow(backbone="resnet18", flow_steps=8)
        model = models_types[model_cls]["class"]()
        
        # Train
        callbacks = models_types[model_cls]["callbacks"]
        
        engine = Engine(
            callbacks=callbacks,
            accelerator="auto",  # \<"cpu", "gpu", "tpu", "ipu", "hpu", "auto">,
            devices=1,
            logger=False,
            #auto_lr_find=True,
            max_epochs=max_epochs
        )
        start_train = time.time()
        try:
            engine.fit(datamodule=datamodule, model=model)
            oom = False
        except Exception as e:
            print("+"*80)
            print(f"Error occured during traing on class {cls}. Trying to continue:\n {e}")
            print("+"*80)
            oom = True
        train_time = time.time() - start_train
        
        # Eval
        start_test = time.time()
        result = engine.test(datamodule=datamodule, model=model)[0]
        inference_time = time.time() - start_test
        
        result["Train_Time"] = train_time
        result["Inference_Time"] = inference_time
        
        result["Model"] = model_cls
        result["Class"] = cls
        result["OOM"] = oom
        result = pd.DataFrame([result])
        results = pd.concat([results, result], ignore_index=True)
        # Write intermediate result
        results.to_csv(f"{prefix}results.csv")
    
results.to_csv(f"{prefix}results.csv")
