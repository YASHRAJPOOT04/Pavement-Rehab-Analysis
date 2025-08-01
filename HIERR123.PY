import matplotlib
matplotlib.use('Agg')
import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import yaml
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy import stats
from openpyxl import load_workbook
from openpyxl.styles import PatternFill
from openpyxl.utils import get_column_letter

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

#=============================================================================================================================================
#=============================================================================================================================================
@dataclass
class ClusteringConfig:
    """
    Configuration class for hierarchical clustering parameters.
    
    Attributes:
        max_clusters: Maximum number of clusters to consider
        linkage: Linkage method ('ward', 'complete', 'average', 'single')
        affinity: Distance metric to use
        figure_width: Width of road visualization plots
        figure_height: Height of road visualization plots
        output_dir: Directory for output files
        min_samples: Minimum samples required for a leaf cluster
    """
    max_clusters: int = 12
    linkage: str = 'ward'
    affinity: str = 'euclidean'
    figure_width: int = 40
    figure_height: int = 4
    output_dir: str = 'output'
    min_samples: int = 2

#=============================================================================================================================================
@dataclass
class ClusteringResults:
    """
    Container for hierarchical clustering results and metrics.
    
    Attributes:
        optimal_n_clusters: Optimal number of clusters determined
        cluster_labels: Array of cluster assignments
        silhouette_scores: List of silhouette scores for different cluster numbers
        db_scores: List of Davies-Bouldin scores for different cluster numbers
        ch_scores: List of Calinski-Harabasz scores for different cluster numbers
        cophenetic_scores: List of cophenetic scores for different cluster numbers
        final_silhouette: Final silhouette score for optimal clustering
        final_db_score: Final Davies-Bouldin score for optimal clustering
        final_ch_score: Final Calinski-Harabasz score for optimal clustering
        linkage_matrix: Linkage matrix for hierarchical clustering
    """
    optimal_n_clusters: int
    cluster_labels: np.ndarray
    silhouette_scores: List[float]
    db_scores: List[float]
    ch_scores: List[float]
    cophenetic_scores: List[float]
    final_silhouette: float
    final_db_score: float
    final_ch_score: float
    linkage_matrix: np.ndarray

@dataclass
class MergeHistory:
    """
    Track the history and details of hierarchical section merges.
    
    Attributes:
        original_cluster: ID of the cluster being merged
        merged_into: ID of the target cluster
        compromise_level: Level of compromise in the merge decision
        similarity_score: Similarity score between merged clusters
        merge_reason: Description of the merge decision
    """
    original_cluster: int
    merged_into: int
    compromise_level: float
    similarity_score: float
    merge_reason: str

@dataclass
class SectionMetrics:
    """
    Store key metrics for a hierarchically clustered road section.
    
    Attributes:
        length: Length of the road section
        homogeneity: Measure of section homogeneity
        continuity: Measure of spatial continuity
        compromise_level: Accumulated compromise from merges
    """
    length: float
    homogeneity: float
    continuity: float
    compromise_level: float
#================================================================================================================================================
#================================================================================================================================================

class DataPreprocessor:
    """Handles data preprocessing for hierarchical clustering"""

    def __init__(self, config: ClusteringConfig):
        self.config = config

    def setup_plot_style(self) -> None:
        """Set up consistent plot styling"""
        # Use a built-in style instead of seaborn
        plt.style.use('default')
        plt.rcParams['figure.figsize'] = [12, 6]
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['axes.labelsize'] = 10
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
        plt.rcParams['lines.linewidth'] = 2
        plt.rcParams['axes.spines.top'] = False
        plt.rcParams['axes.spines.right'] = False

    def clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean column names by removing extra spaces and standardizing format"""
        df.columns = df.columns.str.strip()
        df.columns = df.columns.str.replace(r'\s+', ' ')
        return df

    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load data and ensure Chainage is properly handled"""
        try:
            file_extension = os.path.splitext(file_path)[1].lower()
            if file_extension == '.csv':
                df = pd.read_csv(file_path)
            elif file_extension in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")

            # Clean column names
            df = self.clean_column_names(df)

            # Rename Chainage column if it exists
            if 'Chainage' in df.columns:
                df = df.rename(columns={'Chainage': 'Chainage(km)'})

            # Validate Chainage values are numeric
            if 'Chainage(km)' in df.columns:
                df['Chainage(km)'] = pd.to_numeric(df['Chainage(km)'], errors='coerce')
                if df['Chainage(km)'].isnull().any():
                    raise ValueError("Some Chainage values could not be converted to numbers")

            return df
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            raise

    def detect_outliers(self, df: pd.DataFrame, columns: List[str], method: str = 'iqr') -> Dict[str, np.ndarray]:
        """Detect outliers using either IQR method or Z-score method"""
        outliers = {}

        for column in columns:
            if method == 'iqr':
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outliers[column] = (df[column] < lower_bound) | (df[column] > upper_bound)

            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(df[column]))
                outliers[column] = z_scores > 3

        return outliers

    def handle_outliers(self, df: pd.DataFrame, columns: List[str],
                       method: str = 'iqr', treatment: str = 'clip') -> pd.DataFrame:
        """Handle outliers in the dataset"""
        df_copy = df.copy()
        outliers = self.detect_outliers(df_copy, columns, method)

        for column in columns:
            if treatment == 'clip':
                df_copy[column] = df_copy[column].clip(
                    lower=df_copy[column].mean() - 3 * df_copy[column].std(),
                    upper=df_copy[column].mean() + 3 * df_copy[column].std()
                )
            elif treatment == 'remove':
                df_copy = df_copy[~outliers[column]]
            elif treatment == 'winsorize':
                df_copy[column] = stats.mstats.winsorize(df_copy[column], limits=[0.05, 0.05])

        return df_copy

    def handle_missing_values(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Handle missing values using advanced interpolation techniques"""
        df_copy = df.copy()
        for col in columns:
            if df_copy[col].isnull().any():
                # First try linear interpolation
                df_copy[col] = df_copy[col].interpolate(method='linear', limit_direction='both')
                # If still has NaNs, try polynomial interpolation
                if df_copy[col].isnull().any():
                    df_copy[col] = df_copy[col].interpolate(method='polynomial', order=2)
                # If still has NaNs, use forward fill and then backward fill
                if df_copy[col].isnull().any():
                    df_copy[col].fillna(method='ffill', inplace=True)
                    df_copy[col].fillna(method='bfill', inplace=True)
        return df_copy

    def normalize_data(self, data: pd.DataFrame) -> np.ndarray:
        """Normalize the input data for hierarchical clustering"""
        return data.values

    def preprocess_data(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Complete preprocessing pipeline for hierarchical clustering"""
        logging.info("Starting data preprocessing for hierarchical clustering")
        
        # Handle missing values first
        df = self.handle_missing_values(df, columns)
        
        # Handle outliers
        df = self.handle_outliers(df, columns, method='iqr', treatment='clip')
        
        # Normalize the data
        df[columns] = self.normalize_data(df[columns])
        
        # Sort by Chainage for consistency
        if 'Chainage(km)' in df.columns:
            df = df.sort_values('Chainage(km)')
        
        logging.info("Data preprocessing completed")
        return df

    def validate_data_for_clustering(self, df: pd.DataFrame, columns: List[str]) -> bool:
        """Validate data is suitable for hierarchical clustering"""
        try:
            # Check for sufficient samples
            if len(df) < 2:
                raise ValueError("Not enough samples for clustering")
            
            # Check for numeric data
            non_numeric = [col for col in columns if not np.issubdtype(df[col].dtype, np.number)]
            if non_numeric:
                raise ValueError(f"Non-numeric columns found: {non_numeric}")
            
            # Check for remaining missing values
            missing_values = df[columns].isnull().sum()
            if missing_values.any():
                raise ValueError(f"Missing values found in columns: {missing_values[missing_values > 0]}")
            
            return True
            
        except Exception as e:
            logging.error(f"Data validation failed: {str(e)}")
            raise
#========================================================================================================================================================
def get_section_length_constraints() -> Tuple[Optional[float], Optional[float]]:                                                               #APPROVED
    """
    Get minimum section length constraints from user input.
    Input is in meters, but returns values in kilometers for internal calculations.
    """
    try:
        min_length_input = input("\nEnter minimum section length in meters (or press Enter to skip): ").strip()
        if min_length_input:
            # Convert meters to kilometers
            min_length_km = float(min_length_input) / 1000
            if min_length_km <= 0:
                raise ValueError("Minimum length must be positive")
            print(f"Minimum section length set to: {min_length_input}m ({min_length_km:.3f}km)")
        else:
            min_length_km = None
            print("No minimum section length constraint applied")

        return min_length_km, None  # We're not using max_length currently

    except ValueError as e:
        logging.error(f"Invalid input for section length: {str(e)}")
        raise
#========================================================================================================================================================
class ClusterAnalyzer:                                                                                             #APPROVED
    """Handles hierarchical clustering analysis and optimization"""

    def __init__(self, config: ClusteringConfig):
        self.config = config
        self.validate_config()
        self.visualizer = Visualizer(config)

    def validate_config(self) -> None:
        """Validate configuration parameters for hierarchical clustering"""
        if self.config.max_clusters < 2:
            raise ValueError("max_clusters must be at least 2")
        if self.config.linkage not in ['ward', 'complete', 'average', 'single']:
            raise ValueError("linkage must be one of 'ward', 'complete', 'average', 'single'")
        if self.config.affinity not in ['euclidean', 'manhattan', 'cosine']:
            raise ValueError("affinity must be one of 'euclidean', 'manhattan', 'cosine'")

    def find_optimal_clustering(self, data: np.ndarray, weights: Dict[str, float] = None) -> ClusteringResults:
        """Find optimal number of clusters using hierarchical clustering with multiple metrics."""
        if data.shape[0] < 2:
            raise ValueError("Not enough samples for clustering")

        # Default weights for metrics
        weights = weights or {
            'silhouette': 0.3,
            'davies_bouldin': 0.3,
            'calinski_harabasz': 0.2,
            'cophenetic': 0.2
        }

        metrics = {
            'silhouette': [],
            'davies_bouldin': [],
            'calinski_harabasz': [],
            'cophenetic': []
        }
        
        # Compute linkage matrix once
        linkage_matrix = linkage(data, method=self.config.linkage, metric=self.config.affinity)
        
        best_labels = None
        best_k = None
        best_score = float('-inf')
        
        n_clusters_range = range(2, min(self.config.max_clusters + 1, data.shape[0]))

        print("\nCalculating hierarchical clustering metrics...")
        
        for k in n_clusters_range:
            # For 'ward' linkage, we must use 'euclidean' metric
            if self.config.linkage == 'ward':
                clustering = AgglomerativeClustering(
                    n_clusters=k,
                    linkage=self.config.linkage
                )
            else:
                # For other linkage methods, we can specify the metric
                clustering = AgglomerativeClustering(
                    n_clusters=k,
                    linkage=self.config.linkage,
                    metric=self.config.affinity
                )
            
            labels = clustering.fit_predict(data)
            
            # Calculate clustering metrics
            metrics['silhouette'].append(silhouette_score(data, labels))
            metrics['davies_bouldin'].append(davies_bouldin_score(data, labels))
            metrics['calinski_harabasz'].append(calinski_harabasz_score(data, labels))
            metrics['cophenetic'].append(self._compute_cophenetic_correlation(data, linkage_matrix))
            
            # Calculate weighted score
            score = sum(weights[metric] * metrics[metric][-1] for metric in weights.keys())
            
            if score > best_score:
                best_score = score
                best_k = k
                best_labels = labels

            print(f"Clusters={k}: Silhouette={metrics['silhouette'][-1]:.3f}, "
                  f"DB={metrics['davies_bouldin'][-1]:.3f}, "
                  f"CH={metrics['calinski_harabasz'][-1]:.3f}, "
                  f"Cophenetic={metrics['cophenetic'][-1]:.3f}")

        return ClusteringResults(
            optimal_n_clusters=best_k,
            cluster_labels=best_labels,
            silhouette_scores=metrics['silhouette'],
            db_scores=metrics['davies_bouldin'],
            ch_scores=metrics['calinski_harabasz'],
            cophenetic_scores=metrics['cophenetic'],
            final_silhouette=metrics['silhouette'][best_k-2],
            final_db_score=metrics['davies_bouldin'][best_k-2],
            final_ch_score=metrics['calinski_harabasz'][best_k-2],
            linkage_matrix=linkage_matrix
        )

    def _compute_cophenetic_correlation(self, data: np.ndarray, linkage_matrix: np.ndarray) -> float:
        """Compute cophenetic correlation coefficient"""
        from scipy.cluster.hierarchy import cophenet
        from scipy.spatial.distance import pdist
        
        c, _ = cophenet(linkage_matrix, pdist(data))
        return c
#========================================================================================================================================================

class Visualizer:
    """Handles visualization tasks for hierarchical clustering"""

    def __init__(self, config: ClusteringConfig):
        self.config = config
        self._setup_output_directory()
        self.setup_plot_style()

    def _setup_output_directory(self) -> None:
        """Ensure output directory exists"""
        os.makedirs(self.config.output_dir, exist_ok=True)

    def setup_plot_style(self) -> None:
        """Set up consistent plot styling"""
        # Use a built-in style instead of seaborn
        plt.style.use('default')
        plt.rcParams['figure.figsize'] = [12, 6]
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['axes.labelsize'] = 10
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
        plt.rcParams['lines.linewidth'] = 2
        plt.rcParams['axes.spines.top'] = False
        plt.rcParams['axes.spines.right'] = False

    def plot_hierarchical_metrics(self, clustering_results: ClusteringResults) -> None:
        """Plot hierarchical clustering evaluation metrics"""
        plt.figure(figsize=(15, 10))
        n_clusters_range = range(2, len(clustering_results.silhouette_scores) + 2)

        # Silhouette Score
        plt.subplot(2, 2, 1)
        plt.plot(n_clusters_range, clustering_results.silhouette_scores,
                marker='o', color='#1f77b4', linewidth=2)
        plt.title('Silhouette Score vs Number of Clusters', fontsize=12)
        plt.xlabel('Number of Clusters')
        plt.ylabel('Silhouette Score')
        plt.grid(True, linestyle='--', alpha=0.7)

        # Davies-Bouldin Score
        plt.subplot(2, 2, 2)
        plt.plot(n_clusters_range, clustering_results.db_scores,
                marker='o', color='#2ca02c', linewidth=2)
        plt.title('Davies-Bouldin Score vs Number of Clusters', fontsize=12)
        plt.xlabel('Number of Clusters')
        plt.ylabel('Davies-Bouldin Score')
        plt.grid(True, linestyle='--', alpha=0.7)

        # Calinski-Harabasz Score
        plt.subplot(2, 2, 3)
        plt.plot(n_clusters_range, clustering_results.ch_scores,
                marker='o', color='#ff7f0e', linewidth=2)
        plt.title('Calinski-Harabasz Score vs Number of Clusters', fontsize=12)
        plt.xlabel('Number of Clusters')
        plt.ylabel('Calinski-Harabasz Score')
        plt.grid(True, linestyle='--', alpha=0.7)

        # Cophenetic Score
        plt.subplot(2, 2, 4)
        plt.plot(n_clusters_range, clustering_results.cophenetic_scores,
                marker='o', color='#d62728', linewidth=2)
        plt.title('Cophenetic Score vs Number of Clusters', fontsize=12)
        plt.xlabel('Number of Clusters')
        plt.ylabel('Cophenetic Score')
        plt.grid(True, linestyle='--', alpha=0.7)

        # Add optimal cluster number annotation to each plot
        optimal_n = clustering_results.optimal_n_clusters
        for i in range(4):
            plt.subplot(2, 2, i+1)
            plt.axvline(x=optimal_n, color='gray', linestyle='--', alpha=0.5)
            plt.text(optimal_n, plt.ylim()[0], f'Optimal\n(n={optimal_n})',
                    rotation=90, verticalalignment='bottom')

        plt.tight_layout()
        metrics_path = os.path.join(self.config.output_dir, 'hierarchical_clustering_metrics.png')
        plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
        plt.close()

        logging.info(f"Hierarchical clustering metrics visualization saved to {metrics_path}")

    def create_road_visualization(self, df: pd.DataFrame, hex_colors: List[str]) -> None:
        """Create road visualizations showing hierarchical clustering results"""
        plt.figure(figsize=(self.config.figure_width, self.config.figure_height * 2.2))

        # Only show optimized clusters (remove original clusters subplot)
        ax = plt.subplot(1, 1, 1)
        df_sorted = df.sort_values('Chainage(km)')
        current_cluster = None
        segment_start = None
        prev_chainage = None
        label_positions = []

        # Plot hierarchical clustering segments
        for i, row in df_sorted.iterrows():
            if current_cluster != row['Cluster']:
                if segment_start is not None:
                    cluster_color = f"#{hex_colors[int(current_cluster) % len(hex_colors)]}"
                    ax.plot([segment_start, prev_chainage], [0, 0],
                           linewidth=20,
                           color=cluster_color,
                           solid_capstyle='butt')

                    # Add cluster number with overlap prevention
                    mid_point = (segment_start + prev_chainage) / 2
                    label_y = 0.1

                    while any(abs(pos[0] - mid_point) < 0.5 and abs(pos[1] - label_y) < 0.05
                            for pos in label_positions):
                        label_y += 0.05

                    label_positions.append((mid_point, label_y))
                    ax.text(mid_point, label_y, str(int(current_cluster)),
                           horizontalalignment='center',
                           fontsize=8,
                           bbox=dict(facecolor='white',
                                   edgecolor='none',
                                   alpha=0.7,
                                   pad=1))

                current_cluster = row['Cluster']
                segment_start = row['Chainage(km)']

            prev_chainage = row['Chainage(km)']

        # Draw final segment
        if segment_start is not None:
            cluster_color = f"#{hex_colors[int(current_cluster) % len(hex_colors)]}"
            ax.plot([segment_start, prev_chainage], [0, 0],
                   linewidth=20,
                   color=cluster_color,
                   solid_capstyle='butt')

        # Customize plot
        ax.set_ylim(-0.5, 0.8)
        ax.set_xlabel('Chainage(km)', fontsize=10)
        ax.set_title('Road Sections by Hierarchical Clustering\n' + 
                    f'(Linkage Method: {self.config.linkage}, Affinity: {self.config.affinity})', 
                    fontsize=12, pad=15)
        ax.get_yaxis().set_visible(False)
        ax.grid(True, axis='x', linestyle='--', alpha=0.7)

        # Add section lengths
        length_positions = []
        for i, row in df_sorted.iterrows():
            if row['Section_Start']:
                start_chainage = row['Chainage(km)']
                end_chainage = df_sorted[df_sorted['Section_End'] &
                               (df_sorted['Cluster'] == row['Cluster']) &
                               (df_sorted['Chainage(km)'] >= start_chainage)]['Chainage(km)'].iloc[0]
                section_length = (end_chainage - start_chainage) * 1000
                mid_point = (start_chainage + end_chainage) / 2

                length_y = -0.2
                while any(abs(pos[0] - mid_point) < 0.5 and abs(pos[1] - length_y) < 0.05
                        for pos in length_positions):
                    length_y -= 0.05

                length_positions.append((mid_point, length_y))
                ax.text(mid_point, length_y, f'{section_length:.0f}m',
                       horizontalalignment='center',
                       fontsize=7,
                       bbox=dict(facecolor='white',
                               edgecolor='none',
                               alpha=0.7,
                               pad=1))

        plt.tight_layout(pad=1.5)
        viz_path = os.path.join(self.config.output_dir, 'road_visualization.png')
        plt.savefig(viz_path, bbox_inches='tight', dpi=300)
        plt.close()

        logging.info(f"Road visualization saved to {viz_path}")

    def create_summary_report(self, df: pd.DataFrame, clustering_results: ClusteringResults) -> None:
        """Create a summary report with key findings from hierarchical clustering"""
        report_path = os.path.join(self.config.output_dir, 'clustering_report.txt')

        with open(report_path, 'w') as f:
            f.write("Hierarchical Clustering Analysis Summary Report\n")
            f.write("==========================================\n\n")

            # Initial Clustering Results
            f.write("1. Hierarchical Clustering Results\n")
            f.write("--------------------------------\n")
            f.write(f"Optimal number of clusters: {clustering_results.optimal_n_clusters}\n")
            f.write(f"Final Silhouette Score: {clustering_results.final_silhouette:.3f}\n")
            f.write(f"Final Davies-Bouldin Score: {clustering_results.final_db_score:.3f}\n")
            f.write(f"Final Calinski-Harabasz Score: {clustering_results.final_ch_score:.3f}\n\n")

            # Rest of the report remains the same...

    def create_excel_output(self, df: pd.DataFrame) -> List[str]:
        """
        Create color-coded Excel output for hierarchical clusters.
        """
        # Clean column names before saving
        df = df.copy()
        df.columns = df.columns.str.replace(r'[^\w\s-]', '_')  # Replace special chars with underscore
        
        output_file = os.path.join(self.config.output_dir, "clustered_data.xlsx")
        
        # Ensure required columns exist
        if 'Cluster' not in df.columns:
            raise ValueError("DataFrame must contain 'Cluster' column")
        
        # Save initial DataFrame to Excel
        df.to_excel(output_file, index=False)
        
        # Load workbook for formatting
        wb = load_workbook(output_file)
        ws = wb.active
        
        # Generate colors using matplotlib colormap
        n_clusters = df['Cluster'].nunique()
        cmap = plt.colormaps['tab20']  # Using tab20 for better color distinction
        colors = [cmap(i/n_clusters) for i in range(n_clusters)]
        
        # Convert RGB to HEX colors
        hex_colors = ['{:02X}{:02X}{:02X}'.format(
            int(r*255), 
            int(g*255), 
            int(b*255)
        ) for r, g, b, _ in colors]
        
        # Find and color the cluster column
        cluster_col_idx = None
        for idx, col in enumerate(df.columns, 1):
            if col == 'Cluster':
                cluster_col_idx = idx
                break
        
        if cluster_col_idx is None:
            raise ValueError("Could not find 'Cluster' column")
        
        # Auto-adjust column widths
        for idx, col in enumerate(df.columns, 1):
            max_length = max(
                df[col].astype(str).apply(len).max(),
                len(str(col))
            )
            col_letter = get_column_letter(idx)
            ws.column_dimensions[col_letter].width = max_length + 2
        
        # Apply colors to cells
        for row in ws.iter_rows(min_row=2, max_row=len(df) + 1,
                               min_col=cluster_col_idx,
                               max_col=cluster_col_idx):
            for cell in row:
                if cell.value is not None:
                    try:
                        cluster_id = int(float(cell.value))
                        color = hex_colors[cluster_id % len(hex_colors)]
                        cell.fill = PatternFill(
                            start_color=color,
                            end_color=color,
                            fill_type="solid"
                        )
                    except (ValueError, TypeError) as e:
                        logging.warning(f"Could not process cell value: {cell.value}. Error: {str(e)}")
        
        # Save the formatted workbook
        wb.save(output_file)
        logging.info(f"Hierarchical clustering results saved to {output_file}")
        
        return hex_colors

#========================================================================================================================================================
def run_analysis(input_file: str = None, config_file: str = None):
    """Main analysis function for hierarchical clustering"""
    # Load configuration
    config = ClusteringConfig()
    if config_file and os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config_dict = yaml.safe_load(f)
            config = ClusteringConfig(**config_dict)
        except Exception as e:
            logging.warning(f"Error loading config file: {str(e)}. Using default configuration.")

    # Create output directory
    try:
        os.makedirs(config.output_dir, exist_ok=True)
        logging.info(f"Created output directory: {config.output_dir}")
    except Exception as e:
        logging.error(f"Failed to create output directory: {str(e)}")
        raise

    # Initialize components
    preprocessor = DataPreprocessor(config)
    analyzer = ClusterAnalyzer(config)
    visualizer = Visualizer(config)

    try:
        # If no input file is provided, ask for it
        if not input_file:
            input_file = input("Please enter the path to your dataset file: ")

        # Load and preprocess data
        df = preprocessor.load_data(input_file)

        # Sort DataFrame by Chainage(km)
        df = df.sort_values('Chainage(km)').copy()

        # Get columns for clustering
        print("\nAvailable columns:", df.columns.tolist())
        columns_input = input("\nEnter column names for clustering (comma-separated): ")
        columns_to_use = [col.strip() for col in columns_input.split(',')]

        # Add logging to check data
        print("\nSelected columns:", columns_to_use)
        print("Sample of data before normalization:")
        print(df[columns_to_use].head())

        # Preprocess data
        df = preprocessor.handle_missing_values(df, columns_to_use)
        normalized_data = preprocessor.normalize_data(df[columns_to_use])

        print("\nSample of normalized data:")
        print(normalized_data[:5])

        # Ask for optimal number of clusters
        optimal_clusters_input = input("\nEnter desired number of clusters (or 'NA' to calculate automatically): ").strip()

        print("\nPerforming hierarchical clustering analysis...")
        if optimal_clusters_input.upper() == 'NA':
            # Use automatic method
            clustering_results = analyzer.find_optimal_clustering(normalized_data)
            print(f"\nAutomatically determined optimal number of clusters: {clustering_results.optimal_n_clusters}")
        else:
            try:
                n_clusters = int(optimal_clusters_input)
                if n_clusters < 2:
                    raise ValueError("Number of clusters must be at least 2")

                # Perform hierarchical clustering with specified number of clusters
                clustering = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    linkage=config.linkage,
                    metric=config.affinity
                )
                labels = clustering.fit_predict(normalized_data)

                # Calculate metrics
                silhouette = silhouette_score(normalized_data, labels)
                db_score = davies_bouldin_score(normalized_data, labels)
                ch_score = calinski_harabasz_score(normalized_data, labels)

                clustering_results = ClusteringResults(
                    optimal_n_clusters=n_clusters,
                    cluster_labels=labels,
                    silhouette_scores=[silhouette],
                    db_scores=[db_score],
                    ch_scores=[ch_score],
                    cophenetic_scores=[],
                    final_silhouette=silhouette,
                    final_db_score=db_score,
                    final_ch_score=ch_score,
                    linkage_matrix=linkage(normalized_data, method=config.linkage, metric=config.affinity)
                )

            except ValueError as e:
                print(f"Invalid input: {str(e)}. Falling back to automatic calculation.")
                clustering_results = analyzer.find_optimal_clustering(normalized_data)
                print(f"\nAutomatically determined optimal number of clusters: {clustering_results.optimal_n_clusters}")

        visualizer.plot_hierarchical_metrics(clustering_results)
        print(f"\nOptimal number of clusters: {clustering_results.optimal_n_clusters}")
        print(f"Unique clusters found: {np.unique(clustering_results.cluster_labels)}")

        # Assign clusters to dataframe
        df['Cluster'] = clustering_results.cluster_labels
        df['Optimized_Cluster'] = df['Cluster']

        # Get section constraints and optimize
        min_section_length, max_section_length = get_section_length_constraints()

        print("\nBefore optimization:")
        print(df[['Chainage(km)', 'Cluster', 'Optimized_Cluster']].head())

        df = optimize_road_sections(df, min_section_length, max_section_length)

        print("\nAfter optimization:")
        print(df[['Chainage(km)', 'Cluster', 'Optimized_Cluster']].head())

        print("\nOptimization complete. Preparing output files...")

        # Let user select columns for output
        selected_columns = select_output_columns(df)

        # Export results with selected columns
        export_results(df, config.output_dir, selected_columns)

        print_section_summary(df)

        # Create visualizations
        hex_colors = visualizer.create_excel_output(df)
        visualizer.create_road_visualization(df, hex_colors)
        visualizer.create_summary_report(df, clustering_results)

        # Generate additional outputs
        generate_cluster_legend(hex_colors, config.output_dir)
        export_section_statistics(df, config.output_dir)

        return df

    except Exception as e:
        logging.error(f"Error during analysis: {str(e)}")
        raise

def optimize_road_sections(df: pd.DataFrame, min_length_km: Optional[float] = None, 
                         max_length_km: Optional[float] = None) -> pd.DataFrame:
    """
    Optimize road sections based on length constraints and cluster assignments.
    
    Args:
        df: DataFrame with clustering results
        min_length_km: Minimum section length in kilometers
        max_length_km: Maximum section length in kilometers
    
    Returns:
        DataFrame with optimized sections
    """
    df = df.copy()
    df = df.sort_values('Chainage(km)')
    
    # Initialize section markers
    df['Section_Start'] = False
    df['Section_End'] = False
    
    current_cluster = None
    section_start_idx = 0
    
    for i in range(len(df)):
        if current_cluster != df.iloc[i]['Cluster']:
            if current_cluster is not None:
                # Mark previous section end
                df.iloc[i-1, df.columns.get_loc('Section_End')] = True
                
                # Check and adjust section length if needed
                section_length = df.iloc[i-1]['Chainage(km)'] - df.iloc[section_start_idx]['Chainage(km)']
                
                if min_length_km and section_length < min_length_km:
                    # Merge with previous or next section based on similarity
                    if i > 1:
                        df.iloc[section_start_idx:i, df.columns.get_loc('Optimized_Cluster')] = \
                            df.iloc[section_start_idx-1]['Optimized_Cluster']
            
            # Mark new section start
            df.iloc[i, df.columns.get_loc('Section_Start')] = True
            current_cluster = df.iloc[i]['Cluster']
            section_start_idx = i
    
    # Mark last section end
    df.iloc[-1, df.columns.get_loc('Section_End')] = True
    
    return df

def select_output_columns(df: pd.DataFrame) -> List[str]:
    """
    Let user select columns for output files.
    
    Args:
        df: DataFrame with clustering results
        
    Returns:
        List of selected column names
    """
    print("\nAvailable columns:")
    for i, col in enumerate(df.columns):
        print(f"{i+1}. {col}")
    
    selected = input("\nEnter column numbers to include (comma-separated) or 'all': ").strip()
    
    if selected.lower() == 'all':
        return df.columns.tolist()
    
    try:
        selected_indices = [int(x.strip())-1 for x in selected.split(',')]
        return [df.columns[i] for i in selected_indices]
    except:
        print("Invalid selection. Using all columns.")
        return df.columns.tolist()

def export_results(df: pd.DataFrame, output_dir: str, columns: List[str]) -> None:
    """
    Export clustering results to CSV and Excel files.
    
    Args:
        df: DataFrame with clustering results
        output_dir: Directory for output files
        columns: Columns to include in output
    """
    # Export to CSV
    csv_path = os.path.join(output_dir, 'clustering_results.csv')
    df[columns].to_csv(csv_path, index=False)
    
    # Export to Excel
    excel_path = os.path.join(output_dir, 'clustering_results.xlsx')
    df[columns].to_excel(excel_path, index=False)
    
    print(f"\nResults exported to:")
    print(f"CSV: {csv_path}")
    print(f"Excel: {excel_path}")

def print_section_summary(df: pd.DataFrame) -> None:
    """
    Print summary of road sections after optimization.
    
    Args:
        df: DataFrame with optimized sections
    """
    print("\nRoad Section Summary:")
    print("=====================")
    
    sections = []
    current_start = None
    current_cluster = None
    
    for i, row in df.iterrows():
        if row['Section_Start']:
            current_start = row['Chainage(km)']
            current_cluster = row['Optimized_Cluster']
        elif row['Section_End']:
            sections.append({
                'cluster': current_cluster,
                'start': current_start,
                'end': row['Chainage(km)'],
                'length': row['Chainage(km)'] - current_start
            })
    
    for i, section in enumerate(sections, 1):
        print(f"\nSection {i}:")
        print(f"Cluster: {section['cluster']}")
        print(f"Start: {section['start']:.3f} km")
        print(f"End: {section['end']:.3f} km")
        print(f"Length: {section['length']*1000:.0f} m")

def generate_cluster_legend(hex_colors: List[str], output_dir: str) -> None:
    """
    Generate a legend showing cluster colors.
    
    Args:
        hex_colors: List of hex color codes for clusters
        output_dir: Directory for output files
    """
    plt.figure(figsize=(8, 2))
    for i, color in enumerate(hex_colors):
        plt.plot([0], [0], 'o', color=f'#{color}', label=f'Cluster {i}', markersize=10)
    plt.legend(ncol=len(hex_colors), bbox_to_anchor=(0.5, 0.5), loc='center')
    plt.axis('off')
    
    legend_path = os.path.join(output_dir, 'cluster_legend.png')
    plt.savefig(legend_path, bbox_inches='tight', dpi=300)
    plt.close()

def export_section_statistics(df: pd.DataFrame, output_dir: str) -> None:
    """
    Export statistical summary of road sections.
    
    Args:
        df: DataFrame with optimized sections
        output_dir: Directory for output files
    """
    stats = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for cluster in df['Optimized_Cluster'].unique():
        cluster_data = df[df['Optimized_Cluster'] == cluster]
        
        stats_dict = {
            'Cluster': cluster,
            'Number of Sections': len(cluster_data),
            'Total Length (km)': cluster_data['Chainage(km)'].max() - cluster_data['Chainage(km)'].min()
        }
        
        # Add statistics for numeric columns
        for col in numeric_cols:
            if col not in ['Chainage(km)', 'Cluster', 'Optimized_Cluster']:
                stats_dict.update({
                    f'{col} Mean': cluster_data[col].mean(),
                    f'{col} Std': cluster_data[col].std(),
                    f'{col} Min': cluster_data[col].min(),
                    f'{col} Max': cluster_data[col].max()
                })
        
        stats.append(stats_dict)
    
    # Create DataFrame and export
    stats_df = pd.DataFrame(stats)
    stats_path = os.path.join(output_dir, 'section_statistics.xlsx')
    stats_df.to_excel(stats_path, index=False)
    
    print(f"\nSection statistics exported to: {stats_path}")

if __name__ == "__main__":
    # When running in Jupyter/Colab, just run the analysis
    if 'ipykernel' in sys.modules:
        try:
            result_df = run_analysis()
        except Exception as e:
            logging.error(f"Error in Jupyter/Colab execution: {str(e)}")
            raise
    # When running from command line, use argument parser
    else:
        try:
            import argparse
            parser = argparse.ArgumentParser(description='Hierarchical Clustering Analysis Tool')
            parser.add_argument('--config', type=str, help='Path to configuration file')
            parser.add_argument('--input', type=str, help='Path to input data file')
            parser.add_argument('--min-section-length', type=float, default=0.250,
                              help='Minimum length for road sections in km (default: 0.250)')
            parser.add_argument('--output-dir', type=str, default='output',
                              help='Directory for output files (default: output)')

            args = parser.parse_args()

            # Run the hierarchical clustering analysis
            result_df = run_analysis(args.input, args.config)

        except Exception as e:
            logging.error(f"Error in command-line execution: {str(e)}")
            raise
#Alligator Cracks Severity, Alligator Crack Length, Longitudinal Cracks Severity, Longitudinal Crack Length
#Alligator Cracks Severity, Alligator Crack Length, Longitudinal Cracks Severity, Longitudinal Crack Length, Affected Disintegration (%), Depression, Affected Bleeding(%), Pothole Severity, Affected Pothole(%), No of potholes, Pothole Area, Affected Ravelling(%), Ravelling Area, Affected Patching(%), Edge Break, LHS Drainage Condition, RHS Drainage Condition, LHS Shoulder Condition, RHS Shoulder Condition, Shoving Severity, Spalled Joint, Crack Slab, Poor Cracks, Faillures, Rutting Width
