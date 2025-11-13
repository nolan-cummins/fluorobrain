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

* **Model:** A **U-Net** from the `segmentation-models-pytorch` (smp) library with a **ResNet-34** encoder, pre-trained on **ImageNet**.
* **Dataset:** A custom PyTorch `Dataset` class (`BrainDataset`) that loads an image and its corresponding mask.
* **Data Augmentation:** Uses the `albumentations` library to apply heavy augmentations to the training data, including:
    * Horizontal & Vertical Flips
    * Rotation and Scaling
    * Elastic Transforms
    * Brightness/Contrast Adjustments
    * *Note: Validation data is only resized and normalized.*
* **Loss Function:** **Dice Loss** (`smp.losses.DiceLoss`), which is well-suited for segmentation tasks.
* **Training Loop:**
    1.  Splits the data into 80% training and 20% validation.
    2.  Trains the model using the Adam optimizer.
    3.  Uses a `ReduceLROnPlateau` scheduler to lower the learning rate if the validation loss stops improving.
    4.  Saves the model with the best validation loss to `best_model.pth`.
* **Evaluation:**
    1.  Loads the saved `best_model.pth`.
    2.  Runs inference on test images.
    3.  Calculates the **Dice Score** (percentage of overlap) to measure accuracy.
    4.  Generates a `matplotlib` plot comparing the model's prediction (overlaid on the original image) with the ground-truth mask.

## References:

    [1] A. Arias, L. Manubens-Gil, and M. Dierssen, “Fluorescent transgenic mouse models for whole-brain imaging in health and disease,” Front. Mol. Neurosci., vol. 15, p. 958222, Sep. 2022, doi: 10.3389/fnmol.2022.958222.
