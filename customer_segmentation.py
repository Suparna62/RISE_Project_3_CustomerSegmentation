import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv('customer_data.csv')

# Features to use for clustering
features = df[['Age', 'Income', 'Frequency', 'Spending']]

# Scale the data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Apply KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_features)

# Print cluster counts
print("\nCustomer counts per cluster:")
print(df['Cluster'].value_counts())

# Save clustered data
df.to_csv('customer_data_with_clusters.csv', index=False)

# Plot clusters in 2D
plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x='Income', y='Spending', hue='Cluster', palette='Set1', s=100)
plt.title('Customer Segmentation - Income vs Spending')
plt.xlabel('Income')
plt.ylabel('Spending')
plt.legend(title='Cluster')
plt.tight_layout()
plt.savefig('cluster_plot.png')
plt.show()
