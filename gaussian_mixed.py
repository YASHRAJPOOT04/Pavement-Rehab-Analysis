import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

@dataclass
class ClusteringConfig:
    """Configuration class for GMM clustering parameters."""
    max_clusters: int = 12
    covariance_type: str = 'full'
    init_params: str = 'kmeans'
    n_init: int = 10
    max_iter: int = 100
    random_state: int = 42
    output_dir: str = 'output'

@dataclass
class ClusteringResults:
    """Container for GMM clustering results and metrics."""
    optimal_n_clusters: int
    cluster_labels: np.ndarray
    silhouette_scores: List[float]
    db_scores: List[float]
    ch_scores: List[float]
    bic_scores: List[float]
    aic_scores: List[float]
    final_silhouette: float
    final_db_score: float
    final_ch_score: float
    gmm_model: GaussianMixture

class DataPreprocessor:
    """Handles data preprocessing for GMM clustering"""
    
    def __init__(self):
        self.scaler = StandardScaler()
    
    def preprocess_data(self, data: pd.DataFrame, columns: List[str]) -> np.ndarray:
        """Preprocess data for clustering"""
        try:
            # Handle missing values
            data = data[columns].fillna(data[columns].mean())
            
            # Normalize the data
            normalized_data = self.scaler.fit_transform(data[columns])
            
            return normalized_data
        except Exception as e:
            logging.error(f"Error in data preprocessing: {str(e)}")
            raise

class GaussianMixedClustering:
    """Main class for Gaussian Mixture Model clustering"""
    
    def __init__(self, config: Optional[ClusteringConfig] = None):
        self.config = config or ClusteringConfig()
        self.preprocessor = DataPreprocessor()
    
    def find_optimal_clusters(self, data: np.ndarray, weights: Optional[Dict[str, float]] = None) -> ClusteringResults:
        """Find optimal number of clusters using multiple metrics"""
        if data.shape[0] < 2:
            raise ValueError("Not enough samples for clustering")

        weights = weights or {
            'silhouette': 0.3,
            'davies_bouldin': 0.2,
            'calinski_harabasz': 0.2,
            'bic': 0.15,
            'aic': 0.15
        }

        metrics = {
            'silhouette': [],
            'davies_bouldin': [],
            'calinski_harabasz': [],
            'bic': [],
            'aic': []
        }
        
        best_models = {}
        n_clusters_range = range(2, min(self.config.max_clusters + 1, data.shape[0]))

        for k in n_clusters_range:
            try:
                gmm = GaussianMixture(
                    n_components=k,
                    covariance_type=self.config.covariance_type,
                    init_params=self.config.init_params,
                    n_init=self.config.n_init,
                    max_iter=self.config.max_iter,
                    random_state=self.config.random_state
                )
                
                labels = gmm.fit_predict(data)
                
                metrics['silhouette'].append(silhouette_score(data, labels))
                metrics['davies_bouldin'].append(davies_bouldin_score(data, labels))
                metrics['calinski_harabasz'].append(calinski_harabasz_score(data, labels))
                metrics['bic'].append(gmm.bic(data))
                metrics['aic'].append(gmm.aic(data))
                
                best_models[k] = (gmm, labels)
            except Exception as e:
                logging.error(f"Error in clustering with {k} clusters: {str(e)}")
                raise

        # Find optimal number of clusters
        normalized_metrics = self._normalize_metrics(metrics)
        best_k = self._select_best_k(normalized_metrics, weights, n_clusters_range)
        
        best_model, best_labels = best_models[best_k]
        
        return ClusteringResults(
            optimal_n_clusters=best_k,
            cluster_labels=best_labels,
            silhouette_scores=metrics['silhouette'],
            db_scores=metrics['davies_bouldin'],
            ch_scores=metrics['calinski_harabasz'],
            bic_scores=metrics['bic'],
            aic_scores=metrics['aic'],
            final_silhouette=metrics['silhouette'][best_k-2],
            final_db_score=metrics['davies_bouldin'][best_k-2],
            final_ch_score=metrics['calinski_harabasz'][best_k-2],
            gmm_model=best_model
        )

    def _normalize_metrics(self, metrics: Dict[str, List[float]]) -> Dict[str, np.ndarray]:
        """Normalize metrics to [0,1] range"""
        normalized = {}
        for metric_name, values in metrics.items():
            values = np.array(values)
            if metric_name == 'silhouette':
                normalized[metric_name] = (values + 1) / 2
            else:
                min_val, max_val = np.min(values), np.max(values)
                if max_val > min_val:
                    normalized[metric_name] = (values - min_val) / (max_val - min_val)
                else:
                    normalized[metric_name] = np.zeros_like(values)
                
                if metric_name in ['davies_bouldin', 'bic', 'aic']:
                    normalized[metric_name] = 1 - normalized[metric_name]
        
        return normalized

    def _select_best_k(self, normalized_metrics: Dict[str, np.ndarray], 
                      weights: Dict[str, float], n_clusters_range: range) -> int:
        """Select best number of clusters based on weighted metrics"""
        scores = np.zeros(len(n_clusters_range))
        for metric, weight in weights.items():
            scores += weight * normalized_metrics[metric]
        
        return n_clusters_range[np.argmax(scores)]

    def fit(self, data: pd.DataFrame, columns: List[str]) -> Tuple[np.ndarray, GaussianMixture]:
        """Fit GMM clustering on the data"""
        try:
            processed_data = self.preprocessor.preprocess_data(data, columns)
            results = self.find_optimal_clusters(processed_data)
            return results.cluster_labels, results.gmm_model
        except Exception as e:
            logging.error(f"Error in fitting GMM clustering: {str(e)}")
            raise

def run_clustering(data: pd.DataFrame, columns: List[str], config: Optional[ClusteringConfig] = None) -> Tuple[np.ndarray, GaussianMixture]:
    """Run GMM clustering analysis"""
    try:
        config = config or ClusteringConfig()
        clustering = GaussianMixedClustering(config)
        return clustering.fit(data, columns)
    except Exception as e:
        logging.error(f"Error in running clustering: {str(e)}")
        raise
