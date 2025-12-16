#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


#load data
penguin = pd.read_csv("palmerpenguins_original.csv")
penguin.head


# In[4]:


#remove NAs if there is any
penguin = penguin.dropna()
penguin.head
#333 entries without NAs


# In[5]:


#overview of number of each species
count = penguin["species"].value_counts()
count


# In[6]:


#keep target columns
penguin1 = penguin[["species","bill_length_mm","bill_depth_mm","flipper_length_mm","body_mass_g"]]

#scatterplots of pairs of features of the data
_ = sns.pairplot(penguin1, hue = "species")
plt.savefig("pairwaise.png")


# In[7]:


#PCA of morphological traits
from sklearn.decomposition import PCA
import mpl_toolkits.mplot3d
from sklearn.preprocessing import StandardScaler

#select the numeric features
penguin2 = penguin1[["bill_length_mm","bill_depth_mm","flipper_length_mm","body_mass_g"]]

#scale the features through standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(penguin2)
pca = PCA(n_components = X_scaled.shape[1])
pca.fit(X_scaled)


# In[8]:


#the variance explained by each PC
print(pca.explained_variance_ratio_)


# In[ ]:


#first two PCs explained over 88% of the variance in the data 


# In[9]:


#visualization of the first 3 PCs

fig = plt.figure(1, figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d", elev=-150, azim=110)

species = penguin1["species"]
species = pd.factorize(species)[0]
unique = penguin1["species"].unique()

X_reduced = PCA(n_components = 3).fit_transform(X_scaled)
scatter = ax.scatter(
    X_reduced[:, 0],
    X_reduced[:, 1],
    X_reduced[:, 2],
    c = species,
    s = 30,
)

ax.set(
    title="First three PCA dimensions",
    xlabel="1st PC",
    ylabel="2nd PC",
    zlabel="3rd PC",
)
ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])
ax.zaxis.set_ticklabels([])

# Add a legend
legend1 = ax.legend(
    scatter.legend_elements()[0],
    unique,
    loc="upper right",
    title="Species",
)
ax.add_artist(legend1)

plt.show()


# In[10]:


# visualization of the first 2 PCs

plt.figure()
colors = ["navy", "turquoise", "darkorange"]
lw = 1

for color, i, j in zip(colors, [0, 1, 2], unique):
    plt.scatter(
        X_reduced[species == i, 0], X_reduced[species == i, 1], 
        color = color, alpha = 0.5, lw = lw, label = j
    )

plt.legend(loc="best", shadow=False, scatterpoints=1)
plt.title("PCA of penguin dataset")
plt.xlabel("1st PC")
plt.ylabel("2nd PC")

plt.savefig("pca2d.png")


# In[11]:


#the loadings of each variable on the PCs
loadings_df = pd.DataFrame(
    pca.components_,
    columns=penguin2.columns if hasattr(penguin2, 'columns') else [f'Feature_{i}' for i in range(penguin2.shape[1])],
    index=[f'PC{i+1}' for i in range(pca.components_.shape[0])]
)

print(loadings_df)


# In[ ]:


#PC1 is associated with overall bigger size longer but less deep bills
#PC2 is associated with overall bigger and deeper bills

