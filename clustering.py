import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def perform_clustering(input_file='study_sessions.csv', output_file='clustered_sessions.csv'):
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: {input_file} not found. Please run data_generator.py first.")
        return

    # Features for clustering (including focus_score to define the STATE)
    features = ['session_duration_min', 'break_frequency', 'focus_score', 'time_of_day', 'time_since_last_break']
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[features])
    
    # 3 Clusters: Optimal, Fatigued, Distracted
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['cluster'] = kmeans.fit_predict(scaled_data)
    
    # Analyze clusters to assign meaningful labels
    # We expect:
    # High Focus, Good Duration -> Optimal (0)
    # Low Focus, Long Duration -> Fatigued (1)
    # Variable Focus, Short Duration -> Distracted (2)
    # Note: K-Means labels are arbitrary, we need to map them based on centroids.
    
    cluster_means = df.groupby('cluster')[features].mean()
    print("Cluster Centers:\n", cluster_means)
    
    # Simple heuristic to map clusters to names
    # Sort clusters by focus_score ascending.
    # Lowest focus -> Fatigued/Distracted?
    # Let's map based on focus_score:
    # Highest focus -> Low Fatigue Risk
    # Medium focus -> Moderate Risk
    # Lowest focus -> High Fatigue Risk
    
    sorted_clusters = cluster_means.sort_values('focus_score', ascending=False).index
    
    # Map: 0 (Highest Focus) -> Low Risk, 1 -> Medium, 2 (Lowest) -> High Risk
    risk_mapping = {
        sorted_clusters[0]: 'Low Risk',
        sorted_clusters[1]: 'Medium Risk',
        sorted_clusters[2]: 'High Risk'
    }
    
    df['fatigue_risk'] = df['cluster'].map(risk_mapping)
    
    # Save results
    df.to_csv(output_file, index=False)
    print(f"Clustering complete. Results saved to {output_file}")
    
    # Visualization
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='session_duration_min', y='focus_score', hue='fatigue_risk', palette='viridis')
    plt.title('Fatigue Clusters: Duration vs Focus')
    plt.savefig('cluster_analysis.png')
    print("Visualization saved to cluster_analysis.png") # Changed filename to avoid confusion with script

if __name__ == "__main__":
    perform_clustering()
