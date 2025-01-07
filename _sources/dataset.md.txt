# Dataset

This project utilizes publicly available datasets designed for land cover classification tasks in urban and rural environments. Below is a detailed summary of the datasets, their classes, and primary use cases.

**Source Acknowledgment**
The content for this dataset summary is derived from the official dataset sources listed below. For more detailed and up-to-date information, please refer to the respective dataset pages.

---

**[LandCover Urban/Rural Climate Dataset](https://huggingface.co/datasets/erikpinhasov/landcover_dataset)**
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

### Key Features:

- **Annotations**: Pixel-level segmentation masks aligned with the 8 defined classes.
- **Applications**: Suitable for tasks such as urban planning, environmental monitoring, and rural land cover analysis.

### Use Cases:

1. **Urban Land Cover Analysis**:
   - Analyze urban structures such as roads, buildings, and developed spaces.
2. **Rural Land Cover Monitoring**:
   - Identify agricultural land, water bodies, and natural features like forests and rangelands.

### Dataset Access:

- **Dataset Link**: [LandCover Urban/Rural Climate Dataset](https://huggingface.co/datasets/erikpinhasov/landcover_dataset)
- **Loading Example**:

  ```python
  from datasets import load_dataset

  dataset = load_dataset("erikpinhasov/landcover_dataset")
  print(dataset)
  ```

---

The content in this section has been adapted directly from the [Global Land Cover Mapping (OpenEarthMap)](https://www.kaggle.com/datasets/aletbm/global-land-cover-mapping-openearthmap) Kaggle page. This dataset provides globally distributed land cover data with diverse features and high-quality segmentation annotations, supporting various land cover classification tasks across different geographic regions.
For more details, visit the [official daataset page)](https://www.kaggle.com/datasets/aletbm/global-land-cover-mapping-openearthmap)

#### Key Features:

- **Global Coverage**: Includes images from urban, suburban, and rural settings worldwide.
- **High Resolution**: Suitable for detailed land cover mapping and segmentation.
- **Classes**: Various land cover types, including vegetation, water bodies, and built-up areas.

#### Applications:

1. **Environmental Monitoring**:
   - Analyze the impact of human activities on natural ecosystems globally.
2. **Land Use Planning**:
   - Support urban development strategies and agricultural resource management.
3. **Climate Change Analysis**:
   - Understand the relationship between land cover changes and climate effects.

**Credits**: This dataset is maintained by [Alet BM](https://www.kaggle.com/aletbm) on Kaggle.
