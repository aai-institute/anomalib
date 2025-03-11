# General imports
import numpy as np
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from matplotlib import pyplot as plt
from PIL import Image
from pathlib import Path
from glob import glob
import pandas as pd


# Anomalib imports
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
from anomalib.data.datasets.image.mvtec import CATEGORIES

# TODO: Parameter handling
# Make Sure that prefix relects the used flow layer (currently need to be set 
# using the default parameters set in the all_in_one layer)
rows = 3
cols = 3
max_epochs = 100
CATEGORIES = CATEGORIES
prefix = "original_cflow_"


results = pd.DataFrame()
for cls in CATEGORIES:
    print(f"Training on class {cls}")
    dataset_root = Path.cwd() / "datasets" / "MVTec"
    
    datamodule = MVTec(
            root = dataset_root,
            category=cls,
            train_batch_size=32,
            eval_batch_size=32,
            num_workers=0,
    )
    
    #model = Fastflow(backbone="resnet18", flow_steps=8)
    model = Cflow()
    
    # Train
    callbacks = [
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
    
    engine = Engine(
        callbacks=callbacks,
        accelerator="auto",  # \<"cpu", "gpu", "tpu", "ipu", "hpu", "auto">,
        devices=1,
        logger=False,
        #auto_lr_find=True,
        max_epochs=max_epochs
    )

    try:
        engine.fit(datamodule=datamodule, model=model)
        oom = False
    except Exception as e:
        print(f"Error occured during traing on class {cls}. Trying to continue:\n {e}")
        oom = True
    
    # Eval
    result = engine.test(datamodule=datamodule, model=model)[0]
    result["Class"] = cls
    result["OOM"] = oom
    result = pd.DataFrame([result])
    results = pd.concat([results, result], ignore_index=True)
    
    # Visualize
    test_imgs = glob(str(dataset_root / cls / "test") + "/**/*.png")

    n_sample = rows * cols
    test = np.random.choice(test_imgs, n_sample)
    fig, axes = plt.subplots(rows, 3*cols)

    images = []
    for image_path, ax in zip(test, axes[:,:cols].flatten()):
        image = np.array(Image.open(image_path))
        ax.imshow(image)
        ax.set_xticks([])
        ax.set_yticks([])
        images.append(image)

    preds = []
    for data_path in test:
        prediction = engine.predict(model=model, data_path=data_path)
        preds.append(prediction[0])

    for prediction, ax in zip(preds, axes[:, cols:2*cols].flatten()):
        anomaly_map = prediction.anomaly_map[0]
        anomaly_map = anomaly_map.cpu().numpy().squeeze()
        ax.imshow(anomaly_map)
        ax.set_xticks([])
        ax.set_yticks([])

    for image, prediction, ax in zip(images, preds, axes[:, 2*cols: 3*cols].flatten()):
        anomaly_map = prediction.anomaly_map[0]
        anomaly_map = anomaly_map.cpu().numpy().squeeze()
        if len(image.shape) == 2:
            image = np.repeat(image.reshape(list(image.shape) + [1]), 3, -1)
        heat_map = superimpose_anomaly_map(anomaly_map=anomaly_map, image=image, normalize=True)
        ax.imshow(heat_map)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(f"{prefix}{cls}_visualization.png")
    plt.show()
    
results.to_csv(f"{prefix}results.csv")