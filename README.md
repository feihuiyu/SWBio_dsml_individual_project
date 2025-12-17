# SWBio_dsml_individual_project

This is the individual project of SWBio data science and machine learning teaching unit.

# PCA of morphological traits for distinguishing penguin species

## Introduction

Morphological traits of animals are essential in understanding their behavior, ecological roles and evolution. For example, a large variety of bird species display sexual dimorphism, where the differences in size and color between sexes can be an indicator of sexual selection and survival strategy. Morphological traits are also crucial in distinguishing closely related species. In penguins, variations in bill shape and flipper morphology among species can reflect different adaptation to diet and environmental conditions (Chávez-Hoffmeister, 2020).

In this project, I explored how morphological traits distinguish three penguin species in Antarctica.

## Initial Setup

This project uses uv for package management.

### 1. Clone the repository

``` python
git clone https://github.com/feihuiyu/SWBio_dsml_individual_project.git
cd SWBio_dsml_individual_project
```

### 2. Install dependencies

The libraries used in this project are pandas seaborn scikit-learn numpy matplotlib

Please run the following command to sync the environment:

``` python
uv sync
```

### 3. Launch the code in Jupyter lab

``` python
uv run jupyter lab
```

The full script is included as **penguin_script.py**.

## Methods

### 1.The dataset

The dataset is from a study on ecological sexual dimorphism of Antarctic Penguins (Genus *Pygoscelis*) published by Gorman, Williams and Fraser in 2014.

-   Data file: **palmerpenguins_original.csv**

After removing rows with missing values, the dataset contains 333 observations of three penguin species, including 146 Adelie penguins, 119 Gentoo penguins and 68 Chinstrap penguins. The morphological features measured in this dataset are:

-   **Bill length** (numeric; unit: mm; variable name: bill_length_mm)

-   **Bill depth** (numeric; unit: mm; variable name: bill_depth_mm)

-   **Flipper length** (numeric; unit: mm, variable name: flipper_length_mm)

-   **Body mass** (numeric; unit: g; variable name: body_mass_g)

### 2. Statistical method

Principal component analysis (PCA) is a widely applied machine learning method in studies associated with morphological traits. It reduces the dimensionality of the data, allowing us to identify the key dimensions where the morphological features of the penguins exhibit most variance, providing a helpful tool for identifying potential proxies to distinguish closely related species.

I started exploring the data by plotting pairwise scatter plots of the morphological features.

![pic1](https://github.com/feihuiyu/SWBio_dsml_individual_project/blob/main/pairwaise.png)

Each point on each scatter plot refers to one of the 333 penguins, with the color indicating their respective species. We can see that Gentoo penguin is roughly distinguishable from the other two species by bill shape and flipper length.

I scaled the four morphological features through Z-score normalization and conducted the PCA using the code in cell [7]:

``` python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(penguin2)
pca = PCA(n_components = X_scaled.shape[1])
pca.fit(X_scaled)
```

I calculated the variance explained by each principal component (PC).

I also calculated the loadings of each feature on the PCs using the code in cell [11]:

``` python
loadings_df = pd.DataFrame(
    pca.components_,
    columns=penguin2.columns if hasattr(penguin2, 'columns') else [f'Feature_{i}' for i in range(penguin2.shape[1])],
    index=[f'PC{i+1}' for i in range(pca.components_.shape[0])]
)
```

## Results

The variance in the data explained by each PC is shown below (output of cell[8])

```         
[0.68633893 0.19452929 0.09216063 0.02697115]
```

The first two PCs explained over 88% of the variance in the data, the cumulative variance explained by the first three PCs is over 97%.

![pic2](https://github.com/feihuiyu/SWBio_dsml_individual_project/blob/main/pca2d.png)

This is the output of cell[10]. Each point on the plot refers to a penguin with color indicating its species. With the first 2 PCs, Gentoo penguins are clearly separated from the others. Adelie and Chinstrap penguins are roughly distinguished with a some overlaps.

![pic3](https://github.com/feihuiyu/SWBio_dsml_individual_project/blob/main/pca3d.png)

This is the output of cell[9]. Each point on the 3D space refers to a penguin with color indicating its species. With three PCs, the three penguin species are well separated from each other.

To balance the number of PCs and the amount of variation explained, I will keep PC1 and PC2 as proxies to distinguish the three penguin species.

The loadings of each morphological feature on the PCs is shown below (output of cell[11])

```         
     bill_length_mm  bill_depth_mm  flipper_length_mm  body_mass_g
PC1        0.453753      -0.399047           0.576825     0.549675
PC2        0.600195       0.796170           0.005788     0.076464
PC3        0.642495      -0.425800          -0.236095    -0.591737
PC4       -0.145170       0.159904           0.781984    -0.584686
```

PC1 shows high positive loadings of flipper length, body mass, and bill length, as well as a negative loading on bill depth. Therefore, it is associated with overall bigger size and longer but less deep bills. PC2 has high positive loadings of bill depth and bill length, thus is associated with overall bigger and deeper bills.

## Conclusion

In this project, I performed a PCA on morphological features of three closely related Antarctic penguin species to identify the dimensions that maximize the variance. I found that with the first two PCs explaining over 87% of the variation, Adelie, Gentoo, and Chinstrap penguins can be largely separated. PC1 is associated with body size and shape of the bill, PC2 reflects the overall size of the bill. This can be applied as a preliminary way to identify penguin species when researching them under harsh environmental conditions.
