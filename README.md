In vivo brain experiments using neural probes on mice involve targeting specific regions of the brain for electrical and/or chemical sensing. However, a significant challenge lies in locating the precise location to perform the craniotomy and subsequent angled probe insertion. Furthermore, it is relevant to insert into a set depth within the brain to spatially target specific regions of brain activity. Typically, mice genetically engineered to produce fluorescent proteins in certain parts of the brain are used so that the brain can be observed without histology [1]. However, this fluorescence occurs radially at a certain depth inside the brain, and thus has to radiate through brain matter, dura mater, bone, and skin, greatly obscuring fine details. Any further dust, glue, or hair can similarly complicate observation. This makes it especially difficult to map the cortical region in real-time during surgery.

To address this, we trained a U-Net model using pre-trained ImageNet weights to start, used our dataset for a new decoder, and optimized the entire model to minimize the overlap accuracy (dice score) between the prediction and hand-labeled masks.

## 1. Dataset Preprocessing

The first step is to create and organize the dataset. We used _labelme_ to assign our masks, and then the following algorithm for organization.

* Scans an input directory (`Target\annotation pairs`) for `.json` files.
* For each `.json` file, it:
    1.  Extracts the raw image data embedded within the JSON.
    2.  Reads the polygon/shape annotations.
    3.  Converts these vector shapes into a single pixel-level mask. Each label (e.g., "brain") is assigned an integer value (e.g., 1), and the background is 0.
* Saves the files into two new directories:
    * **`Target\dataset\images`**: The original image (e.g., `my_image.png`).
    * **`Target\dataset\SegmentationClass`**: The corresponding segmentation mask (e.g., `my_image.png`).

## 2. Model Training & Evaluation

This pipeline is implemented in the **`run model.ipynb`** notebook.

* **Model Architecture:** A **U-Net** from the `segmentation-models-pytorch` (smp) library with an **EfficientNet-B0** encoder, pre-trained on **ImageNet**.
* **Specialization:** Two independent models are trained to maximize sensitivity: one for Region 1 (SSp) and one for Region 2 (VISp).
* **Dataset:** A custom PyTorch `Dataset` class (`BrainDataset`) handles image/mask loading.
* **Data Augmentation:** Uses the `albumentations` library to prevent overfitting on the small dataset ($N=71$ training), applying:
    * Horizontal & Vertical Flips
    * Rotation and Scaling
    * Elastic Transformations
    * Brightness/Contrast Adjustments
* **Loss Function:** Combined **Dice Loss** (`smp.losses.DiceLoss`) and Binary Cross-Entropy (BCE).
* **Training Loop:**
    1.  Splits data into training (80%) and validation (20%).
    2.  Uses **Gradient Accumulation** to simulate larger batch sizes on memory-constrained hardware.
    3.  Optimizes using Adam with a `ReduceLROnPlateau` scheduler.
    4.  Saves the weights with the best validation Dice score to `best_model.pth`.

## 3. Geometric Alignment Algorithm

This post-processing pipeline is implemented in the **`alignment and analysis.ipynb`** notebook.

* **Objective:** Register a ground-truth vector map onto the "blobby" model predictions to enforce biological constraints (fixed distance between regions).
* **Algorithm:** **Multi-Scale Gaussian Pyramid Search** ($L=3$ levels).
* **Optimization Parameters:** Scale ($s$), Rotation ($\theta$), and Translation ($t_x, t_y$).
* **Process:**
    1.  **Coarse-to-Fine Search:** Iteratively sweeps scale and rotation parameters at decreasingly granular steps.
    2.  **Phase Correlation:** Computes optimal translation shift in the frequency domain for sub-pixel accuracy.
    3.  **Scoring:** Optimizes a weighted Intersection-over-Union (IoU) metric that penalizes non-overlapping regions to lock the vector map into the correct anatomical pose.
* **Analysis:** Calculates Dice scores for (1) Model vs. GT, (2) Model vs. Aligned Map, and (3) GT vs. Aligned Map.

## 4. Results
<img width="4862" height="2292" alt="532_result" src="https://github.com/user-attachments/assets/820a1c84-7fc2-40c1-9340-ad00cbff383a" />


## References

    [1] A. Arias, L. Manubens-Gil, and M. Dierssen, “Fluorescent transgenic mouse models for whole-brain imaging in health and disease,” Front. Mol. Neurosci., vol. 15, p. 958222, Sep. 2022, doi: 10.3389/fnmol.2022.958222.
