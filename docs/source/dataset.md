# Dataset

This project utilizes publicly available datasets designed for land cover classification tasks in urban and rural environments. Below is a detailed summary of the datasets, their classes, and primary use cases.

```{note}
The content for this dataset summary is derived from the official dataset sources listed below. For more detailed and up-to-date information, please refer to the source [Alet BM](https://www.kaggle.com/datasets/aletbm/global-land-cover-mapping-openearthmap).
```

---

## [LandCover Urban / Rural Climate Dataset](https://huggingface.co/datasets/erikpinhasov/landcover_dataset)

This dataset provides comprehensive data for land cover classification, focusing on both urban and rural regions. The dataset includes high-resolution imagery and annotations mapped to the following **8 classes**:

### Classes (`id2label` Mapping):

| Class ID | Class Name       |
| -------- | ---------------- |
| 0        | Background       |
| 1        | Bareland         |
| 2        | Rangeland        |
| 3        | Developed Space  |
| 4        | Road             |
| 5        | Tree             |
| 6        | Water            |
| 7        | Agriculture Land |
| 8        | Buildings        |

---

### Preprocessing and Modifications

The original dataset from [Alet BM](https://www.kaggle.com/datasets/aletbm/global-land-cover-mapping-openearthmap) contained images with a resolution of **1000 × 1000 pixels**. To prepare the data for training and improve model performance:

- The images were split into **512 × 512 tiles**.
- A **24-pixel overlap** was applied between adjacent tiles to ensure no loss of contextual information at the edges.
- Corresponding segmentation masks were also tiled to match the new image sizes and maintain alignment.

These modifications ensure compatibility with the segmentation model and allow better handling of edge cases during training and inference.

---

### Use Cases:

1. **Urban Land Cover Analysis**:
   - Analyze urban structures such as roads, buildings, and developed spaces.
2. **Rural Land Cover Monitoring**:
   - Identify agricultural land, water bodies, and natural features like forests and rangelands.

---

**Loading Example:**

```python
from datasets import load_dataset

dataset = load_dataset("erikpinhasov/landcover_dataset")
print(dataset)
```

**HuggingFace**: [LandCover / Rural Climate Dataset](https://huggingface.co/datasets/erikpinhasov/landcover_dataset)

---

### Credits:

This dataset is adapted from [Alet BM](https://www.kaggle.com/datasets/aletbm/global-land-cover-mapping-openearthmap) on Kaggle. The original dataset has been updated and enhanced for improved usability, including modifications to labels, preprocessing steps, and documentation to better support land cover classification tasks.
