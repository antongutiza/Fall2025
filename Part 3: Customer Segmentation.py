# Data source: Online Retail Dataset from Kaggle (https://www.kaggle.com/datasets/yasserh/customer-segmentation-dataset)

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

df.dropna(subset=['CustomerID'], inplace=True)
df['CustomerID'] = df['CustomerID'].astype(int)

# Remove cancelled orders (transactions with a negative quantity)
df = df[df['Quantity'] > 0]

# Feature Engineering
# Create a 'Total_Spending' column for each transaction
df['Total_Spending'] = df['Quantity'] * df['UnitPrice']

# Aggregate data by customer to get the required features
customer_data = df.groupby('CustomerID').agg(
    annual_spending=('Total_Spending', 'sum'),
    purchase_frequency=('InvoiceNo', 'nunique')
).reset_index()

# Select numerical features for clustering
features = ['annual_spending', 'purchase_frequency']
X = customer_data[features]

# Scale the numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine optimal number of clusters using the elbow method
inertia = []
K = range(1, 10) # Test a reasonable range for K
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot the elbow curve and save the image
plt.figure(figsize=(8, 5))
plt.plot(K, inertia, 'bo-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.savefig('elbow_plot.png')
plt.close()
print("Elbow plot saved as elbow_plot.png")

optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
customer_data['cluster'] = kmeans.fit_predict(X_scaled)

# Analyze clusters
cluster_summary = customer_data.groupby('cluster')[features].mean().round(2)
print("\n--- Cluster Characteristics ---")
print(cluster_summary)

# Example of targeted strategies
for cluster in range(optimal_k):
    print(f"\nCluster {cluster} Strategy:")
    avg_spending = cluster_summary.loc[cluster, 'annual_spending']
    avg_frequency = cluster_summary.loc[cluster, 'purchase_frequency']

    if avg_spending > 5000:
        print("High-spending, high-frequency customers: Offer exclusive loyalty programs and personalized recommendations.")
    elif avg_spending > 1000 and avg_frequency > 10:
        print("Frequent buyers: Provide early access to sales and bulk discounts.")
    else:
        print("Lower-value customers: Launch re-engagement campaigns with special discounts or free shipping offers.")

# Save cluster assignments to a new CSV file
customer_data.to_csv('customer_segments.csv', index=False)
print("\nCustomer segments saved to customer_segments.csv")
