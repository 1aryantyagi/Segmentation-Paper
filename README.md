# Hybrid Loss-Driven Framework for Automated Parotid Segmentation in Head and Neck CT

![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10%2B-orange)

This repository implements a deep learning framework for automatic parotid gland segmentation in head and neck CT scans using 3D U-Net architectures with a novel hybrid loss function. The approach addresses class imbalance and boundary inaccuracy challenges in medical image segmentation.
Link - https://journals.lww.com/jomp/fulltext/2025/07000/a_combined_loss_driven_framework_for_automated.13.aspx

🔑 Key Features
---------------

*   **Two Architectures**:
    
    *   Residual 3D U-Net with Layer Normalization
        
    *   Attention-Augmented 3D U-Net (CBAM-enhanced)
        
*   **Hybrid Loss Function**: 0.7 × mDSC + 0.3 × Focal Loss
    
*   **Advanced Training**:
    
    *   Custom checkpointing (peak DSC + min validation loss)
        
    *   Learning rate scheduling
        
    *   Volumetric CT processing
        
*   **Comprehensive Metrics**: DSC, IoU, Bland-Altman analysis
    

⚙️ Installation
---------------

```bash
    git clone https://github.com/yourusername/parotid-segmentation.git
    cd parotid-segmentation
    pip install -r requirements.txt
```

**Requirements**:

*   Python 3.8+
    
*   TensorFlow 2.10+
    
*   nibabel, scikit-image, scikit-learn, matplotlib, pandas
    

📁 Dataset Preparation
----------------------

### Data Structure

Organize your dataset as follows:

```text
    data/
    ├── images/
    │   ├── patient1.nii.gz
    │   └── patient2.nii.gz
    ├── masks/
    │   ├── patient1.nii.gz
    │   └── patient2.nii.gz
    └── splits.json
```

### Preprocessing Steps

1.  **Convert DICOM to NIfTI** (using tools like dcm2niix)
    
2.  **Resample to isotropic resolution** (1×1×1 mm³)
    
3.  **Normalize intensity** (window: \[-200, 300 HU\])
    
4.  **Split data** (70% train, 15% validation, 15% test)
    

🚀 Training Models
------------------

Two Jupyter notebooks are provided:

1.  3D_Unet_Parotid.ipynb - Baseline Unet model
    
2.  Attention_Unet.ipynb - CBAM-enhanced model
    

### Training Configuration

**Parameters** 

| Parameter           | Value                         |
|---------------------|-------------------------------|
| Batch Size          | 100                           |
| Initial LR          | 0.001                         |
| Epochs              | 50                            |
| Patch Size          | 128×128×64                    |
| Loss Weights        | α = 0.7 (mDSC), β = 0.3 (FL)  |

**Focal Loss Parameters**:

*   γ=1.0, α=\[0.3 (background), 0.35 (right parotid), 0.35 (left parotid)\]
    

### Custom Checkpointing

Models are saved when **both** conditions are met:

1.  Highest validation DSC
    
2.  Lowest validation loss
    

```python
    class HybridCheckpoint(tf.keras.callbacks.Callback):
        def __init__(self, filepath):
            super().__init__()
            self.filepath = filepath
            self.best_dsc = -np.Inf
            self.best_loss = np.Inf
            
        def on_epoch_end(self, epoch, logs=None):
            current_dsc = logs.get('val_dice_coef')
            current_loss = logs.get('val_loss')
            
            if current_dsc > self.best_dsc and current_loss < self.best_loss:
                self.best_dsc = current_dsc
                self.best_loss = current_loss
                self.model.save(self.filepath, overwrite=True)
```

📊 Evaluation
-------------

### Quantitative Metrics

```python
    def dice_coef(y_true, y_pred):
        intersection = tf.reduce_sum(y_true * y_pred)
        return (2. * intersection + 1e-6) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + 1e-6)
        
    def iou(y_true, y_pred):
        intersection = tf.reduce_sum(y_true * y_pred)
        union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
        return (intersection + 1e-6) / (union + 1e-6)  
```

### Performance Summary

1.  U-Net model

| Organ          | Metric        | Model A    | Model B    | A - B       |
|----------------|---------------|------------|------------|-------------|
| Parotid Left   | Dice Mean     | 0.854631   | 0.865762   | -0.011132   |
| Parotid Left   | Dice Std      | 0.059425   | 0.066597   | -0.007172   |
| Parotid Left   | Dice Median   | 0.868868   | 0.883497   | -0.014628   |
| Parotid Left   | IoU Mean      | 0.750658   | 0.768616   | -0.017958   |
| Parotid Left   | IoU Std       | 0.086854   | 0.091319   | -0.004465   |
| Parotid Left   | IoU Median    | 0.768141   | 0.791307   | -0.023166   |
| Parotid Right  | Dice Mean     | 0.844395   | 0.848335   | -0.003939   |
| Parotid Right  | Dice Std      | 0.097649   | 0.102393   | -0.004744   |
| Parotid Right  | Dice Median   | 0.858146   | 0.870922   | -0.012776   |
| Parotid Right  | IoU Mean      | 0.739939   | 0.746702   | -0.006763   |
| Parotid Right  | IoU Std       | 0.111581   | 0.116147   | -0.004565   |
| Parotid Right  | IoU Median    | 0.751538   | 0.771361   | -0.019823   |

2.  U-Net + Attention

| Organ          | Metric        | Model A    | Model B    | A - B       |
|----------------|---------------|------------|------------|-------------|
| Parotid Left   | Dice Mean     | 0.848844   | 0.861770   | -0.012926   |
| Parotid Left   | Dice Std      | 0.075513   | 0.057465   | 0.018048    |
| Parotid Left   | Dice Median   | 0.865655   | 0.872407   | -0.006752   |
| Parotid Left   | IoU Mean      | 0.743894   | 0.761270   | -0.017376   |
| Parotid Left   | IoU Std       | 0.100083   | 0.082593   | 0.017489    |
| Parotid Left   | IoU Median    | 0.763136   | 0.773694   | -0.010558   |
| Parotid Right  | Dice Mean     | 0.836777   | 0.842324   | -0.005547   |
| Parotid Right  | Dice Std      | 0.099615   | 0.112922   | -0.013307   |
| Parotid Right  | Dice Median   | 0.857330   | 0.869218   | -0.011888   |
| Parotid Right  | IoU Mean      | 0.728953   | 0.739847   | -0.010895   |
| Parotid Right  | IoU Std       | 0.114149   | 0.128088   | -0.013939   |
| Parotid Right  | IoU Median    | 0.750287   | 0.768689   | -0.018402   |


📦 Pretrained Models
--------------------

Download our best-performing models:

*   [3D U-Net with Hybrid Loss](https://drive.google.com/your_model_link)
    
*   [Attention U-Net with Hybrid Loss](https://drive.google.com/your_attention_model_link)
    

Load for inference:

```python
    model = tf.keras.models.load_model('path/to/model.h5', custom_objects={'dice_coef': dice_coef})
```

📈 Results Visualization
------------------------

Comparison of segmentation boundaries using different loss functions_

Bland-Altman analysis showing improved agreement with hybrid loss_

📜 Citation
-----------

If you use this work, please cite:

```bibtex

```

📧 Contact
----------

For questions or collaborations:

*   Aryan Tyagi: [1aryantyagi@email.com](https://mailto:aryan.tyagi@email.com/)
