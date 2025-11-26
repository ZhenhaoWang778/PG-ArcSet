# PG-ArcSet: Arc Image Dataset for Power Grid Fault Detection

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Framework](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)
                               

**Official repository for the paper: "Arc Image Dataset for Power Grid Fault Detection (PG-ArcSet)"**

## 📖 Introduction

**PG-ArcSet** is a standardized, high-voltage arc image dataset designed to facilitate deep learning research in power system fault diagnosis. The dataset captures the dynamic morphological characteristics of arcs under controlled laboratory conditions, addressing the lack of well-annotated public datasets in this field.

The dataset includes **1,950 images** preprocessed to 224×224 pixels, covering three fundamental arc types:
1.  **Line-type Arc:** High-current plasma channel, typical in short-circuit faults.
2.  **Branch-type Arc:** Main branch with divergence, typical in corona discharge.
3.  **Net-type Arc:** Mesh diffusion, typical in wet pollution on insulators.

<!-- [PLACEHOLDER] Upload 'Fig 3. Sample arcs data' from your paper here and name it assets/samples.png -->
![Sample Arcs](assets/samples.png)
*Figure: Samples of Line-type, Branch-type, and Net-type arcs.*

## 📂 Dataset Structure

The dataset is organized into three subfolders corresponding to the arc types. Each image has been centered and resized to 224×224 pixels.

```text
PG-ArcSet/
├── data.zip/ 
│   ├── Line/      # Images of Line-type arcs
│   ├── Branch/    # Images of Branch-type arcs
│   └── Net/       # Images of Net-type arcs
```

