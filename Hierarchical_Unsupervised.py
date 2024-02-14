# Unsupervised
import pandas as pd
import sqlite3
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

filename = os.path.dirname(__file__) + "/" + 'cleaned_movie_earnings_year_fixed.db'
filename = filename.replace("/", "\\")
conn = sqlite3.connect(filename)
#GET TABLE
query = "SELECT * FROM MOVIE_COMBINED"
df = pd.read_sql_query(query, conn)

# Close the database connection
conn.close()

# Select the features for clustering
# values are in x label, using movie features like domestic, worldwide and rating to cluster the movies together
features = df[['Domestic', 'Worldwide', 'Rating']]

# Perform hierarchical clustering
# using the Ward linkage method, which is a method for measuring the dissimilarity between clusters
# values are printed in ylabel
linked = linkage(features, 'ward')

# Plot the dendrogram
dendrogram(linked,
           orientation='top',
           labels=np.arange(len(df)),
           distance_sort='descending',
           show_leaf_counts=True)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Movie Indexes')
plt.ylabel('Distance between Clusters')
plt.show()


# Determine the optimal number of clusters
max_d = 3000000000  # Maximum distance for forming clusters
clusters = fcluster(linked, max_d, criterion='distance')

# Add the cluster labels to the dataframe
df['Cluster'] = clusters

# Display the clustered data
print(df[['Domestic', 'Worldwide', 'Rating', 'Cluster']])


