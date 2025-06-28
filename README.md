# Hybrid Loss-Driven Framework for Automated Parotid Segmentation in Head and Neck CT

[https://img.shields.io/badge/License-MIT-yellow.svg](https://img.shields.io/badge/License-MIT-yellow.svg)

[https://img.shields.io/badge/Python-3.8%2B-blue](https://img.shields.io/badge/Python-3.8%2B-blue)

[https://img.shields.io/badge/TensorFlow-2.10%2B-orange](https://img.shields.io/badge/TensorFlow-2.10%2B-orange)

This repository implements a deep learning framework for automatic parotid gland segmentation in head and neck CT scans using 3D U-Net architectures with a novel hybrid loss function. The approach addresses class imbalance and boundary inaccuracy challenges in medical image segmentation.

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
        
*   **Comprehensive Metrics**: DSC, IoU, Precision, Recall, Bland-Altman analysis
    

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

| Model            | Loss   | Left Parotid DSC (Median)  | Right Parotid DSC (Median)  | Mean IoU |
|------------------|--------|----------------------------|-----------------------------|----------|
| 3D U-Net         | Hybrid | 0.8835                     | 0.8709                      | 0.7515   |
| Attention U-Net  | Hybrid | 0.8724                     | 0.8692                      | 0.7515   |
| 3D U-Net         | CCE    | 0.8657                     | 0.8573                      | 0.7321   |


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