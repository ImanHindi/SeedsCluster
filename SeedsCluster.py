import pandas as pd

# load the training dataset
#!wget https://raw.githubusercontent.com/MicrosoftDocs/mslearn-introduction-to-machine-learning/main/Data/ml-basics/seeds.csv
data = pd.read_csv('seeds.csv')

# Display a random sample of 10 observations (just the features)
features = data[data.columns[0:6]]
features.sample(10)
print(features[data.columns[0:6]])
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA



scaled_features=MinMaxScaler().fit_transform(features)

features_2d=PCA(n_components=2).fit_transform(scaled_features)
print(features_2d[0:10])


import matplotlib.pyplot as plt 

plt.scatter(features_2d[:,0],features_2d[:,1])
plt.xlabel('Dimension1')
plt.ylabel('Dimension2')
plt.title('Data')
plt.show()


import numpy as np
from sklearn.cluster import KMeans


wcss=[]

for i in range(1,11):
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(features.values)
    wcss.append(kmeans.inertia_)


plt.plot(range(1,11),wcss)
plt.title('WCSS by Clusters')
plt.xlabel('K')
plt.ylabel('WCSS')
plt.show()



from sklearn.cluster import KMeans

# Create a model based on 3 centroids
model = KMeans(n_clusters=3, init='k-means++', n_init=100, max_iter=1000)
# Fit to the data and predict the cluster assignments for each data point
km_clusters = model.fit_predict(features.values)
# View the cluster assignments
print(km_clusters)


import matplotlib.pyplot as plt

def plot_clusters(samples, clusters):
    col_dic = {0:'blue',1:'green',2:'orange'}
    mrk_dic = {0:'*',1:'x',2:'+'}
    colors = [col_dic[x] for x in clusters]
    markers = [mrk_dic[x] for x in clusters]
    for sample in range(len(clusters)):
        plt.scatter(samples[sample][0], samples[sample][1], color = colors[sample], marker=markers[sample], s=100)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('Assignments')
    plt.show()

plot_clusters(features_2d, km_clusters)

seed_species = data[data.columns[7]]
plot_clusters(features_2d, seed_species.values)





# clustering the seeds data using an agglomerative clustering algorithm.
from sklearn.cluster import AgglomerativeClustering

agg_model = AgglomerativeClustering(n_clusters=3)
agg_clusters = agg_model.fit_predict(features.values)
print(agg_clusters)


import matplotlib.pyplot as plt


def plot_clusters(samples, clusters):
    col_dic = {0:'blue',1:'green',2:'orange'}
    mrk_dic = {0:'*',1:'x',2:'+'}
    colors = [col_dic[x] for x in clusters]
    markers = [mrk_dic[x] for x in clusters]
    for sample in range(len(clusters)):
        plt.scatter(samples[sample][0], samples[sample][1], color = colors[sample], marker=markers[sample], s=100)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('Assignments')
    plt.show()
