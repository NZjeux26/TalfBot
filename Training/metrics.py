import torch
import numpy as np
from torch.nn import functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
from typing import Tuple, Dict, List
from model import PolicyValueNetwork

class ModelEvaluator:
    def __init__(self, model, device):
        """
        Initialize the evaluator with a trained model and device.
        
        Args:
            model: The trained PolicyValueNetwork model
            device: The device to run evaluations on (cuda/cpu)
        """
        self.model = model
        self.device = device
        self.model.eval()
        
    def compute_metrics(self, dataloader) -> Dict:
        """
        Compute comprehensive metrics for both policy and value predictions.
        
        Args:
            dataloader: PyTorch DataLoader containing validation/test data
            
        Returns:
            Dictionary containing all computed metrics
        """
        # Initialize metric accumulators
        metrics = {
            'policy_metrics': {
                'start_accuracy': 0.0,
                'end_accuracy': 0.0,
                'top_k_accuracy': {k: 0.0 for k in [1, 3, 5]},
                'start_position_entropy': 0.0,
                'end_position_entropy': 0.0,
            },
            'value_metrics': {
                'mse': 0.0,
                'mae': 0.0,
                'correlation': 0.0,
                'value_calibration': []
            },
            'combined_metrics': {
                'total_loss': 0.0,
                'move_pair_accuracy': 0.0
            }
        }
        
        all_value_preds = []
        all_value_targets = []
        total_samples = 0
        
        with torch.no_grad():
            for X_batch, y_start_batch, y_end_batch, y_winner_batch in dataloader:
                # Move data to device
                X_batch = X_batch.to(self.device)
                y_start_batch = y_start_batch.to(self.device)
                y_end_batch = y_end_batch.to(self.device)
                y_winner_batch = y_winner_batch.to(self.device)
                
                # Get model predictions
                y_start_pred, y_end_pred, value_pred = self.model(X_batch)
                
                # Convert one-hot labels to indices
                y_start_indices = y_start_batch.view(y_start_batch.size(0), -1).argmax(dim=1)
                y_end_indices = y_end_batch.view(y_end_batch.size(0), -1).argmax(dim=1)
                
                # Update metrics
                self._update_policy_metrics(
                    metrics['policy_metrics'],
                    y_start_pred, y_end_pred,
                    y_start_indices, y_end_indices
                )
                
                self._update_value_metrics(
                    metrics['value_metrics'],
                    value_pred, y_winner_batch
                )
                
                # Store predictions for later analysis
                all_value_preds.extend(value_pred.cpu().numpy())
                all_value_targets.extend(y_winner_batch.cpu().numpy())
                
                total_samples += X_batch.size(0)
        
        # Normalize accumulated metrics
        self._normalize_metrics(metrics, total_samples)
        
        # Compute correlation for value predictions
        metrics['value_metrics']['correlation'] = np.corrcoef(
            np.array(all_value_preds).flatten(),
            np.array(all_value_targets).flatten()
        )[0, 1]
        
        return metrics
    
    def _update_policy_metrics(self, metrics: Dict, y_start_pred: torch.Tensor,
                             y_end_pred: torch.Tensor, y_start_true: torch.Tensor,
                             y_end_true: torch.Tensor) -> None:
        """Update metrics related to policy predictions."""
        # Accuracy metrics
        metrics['start_accuracy'] += (y_start_pred.argmax(dim=1) == y_start_true).float().sum().item()
        metrics['end_accuracy'] += (y_end_pred.argmax(dim=1) == y_end_true).float().sum().item()
        
        # Top-k accuracy
        for k in metrics['top_k_accuracy'].keys():
            start_topk = torch.topk(y_start_pred, k=k, dim=1).indices
            end_topk = torch.topk(y_end_pred, k=k, dim=1).indices
            
            metrics['top_k_accuracy'][k] += (
                (start_topk == y_start_true.unsqueeze(1)).any(dim=1) &
                (end_topk == y_end_true.unsqueeze(1)).any(dim=1)
            ).float().sum().item()
        
        # Distribution entropy
        metrics['start_position_entropy'] += self._compute_entropy(y_start_pred).sum().item()
        metrics['end_position_entropy'] += self._compute_entropy(y_end_pred).sum().item()
    
    def _update_value_metrics(self, metrics: Dict, value_pred: torch.Tensor,
                            value_true: torch.Tensor) -> None:
        """Update metrics related to value predictions."""
        metrics['mse'] += F.mse_loss(value_pred, value_true, reduction='sum').item()
        metrics['mae'] += F.l1_loss(value_pred, value_true, reduction='sum').item()
        
        # Store predictions for calibration analysis
        metrics['value_calibration'].extend(zip(
            value_pred.cpu().numpy(),
            value_true.cpu().numpy()
        ))
    
    def _normalize_metrics(self, metrics: Dict, total_samples: int) -> None:
        """Normalize accumulated metrics by the total number of samples."""
        # Normalize policy metrics
        metrics['policy_metrics']['start_accuracy'] /= total_samples
        metrics['policy_metrics']['end_accuracy'] /= total_samples
        
        for k in metrics['policy_metrics']['top_k_accuracy'].keys():
            metrics['policy_metrics']['top_k_accuracy'][k] /= total_samples
        
        metrics['policy_metrics']['start_position_entropy'] /= total_samples
        metrics['policy_metrics']['end_position_entropy'] /= total_samples
        
        # Normalize value metrics
        metrics['value_metrics']['mse'] /= total_samples
        metrics['value_metrics']['mae'] /= total_samples
    
    @staticmethod
    def _compute_entropy(probs: torch.Tensor) -> torch.Tensor:
        """Compute entropy of probability distributions."""
        return -(probs * torch.log(probs + 1e-8)).sum(dim=1)
    
    def plot_metrics(self, metrics: Dict, save_path: str = None) -> None:
        """
        Create visualizations of the model's performance metrics.
        
        Args:
            metrics: Dictionary of computed metrics
            save_path: Optional path to save the plots
        """
        plt.figure(figsize=(15, 10))
        
        # Plot policy metrics
        plt.subplot(2, 2, 1)
        accuracies = [
            metrics['policy_metrics']['start_accuracy'],
            metrics['policy_metrics']['end_accuracy']
        ]
        plt.bar(['Start Position', 'End Position'], accuracies)
        plt.title('Position Prediction Accuracy')
        plt.ylabel('Accuracy')
        
        # Plot top-k accuracies
        plt.subplot(2, 2, 2)
        top_k = list(metrics['policy_metrics']['top_k_accuracy'].keys())
        top_k_values = list(metrics['policy_metrics']['top_k_accuracy'].values())
        plt.plot(top_k, top_k_values, marker='o')
        plt.title('Top-K Accuracy')
        plt.xlabel('K')
        plt.ylabel('Accuracy')
        
        # Plot value prediction correlation
        plt.subplot(2, 2, 3)
        value_preds = [x[0] for x in metrics['value_metrics']['value_calibration']]
        value_true = [x[1] for x in metrics['value_metrics']['value_calibration']]
        plt.scatter(value_true, value_preds, alpha=0.1)
        plt.plot([-1, 1], [-1, 1], 'r--')  # Perfect prediction line
        plt.title(f'Value Prediction Correlation: {metrics["value_metrics"]["correlation"]:.3f}')
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        
        # Plot value calibration
        plt.subplot(2, 2, 4)
        bins = np.linspace(-1, 1, 20)
        pred_binned = np.digitize(value_preds, bins) - 1
        true_means = [np.mean(np.array(value_true)[pred_binned == i]) 
                     for i in range(len(bins)-1)]
        plt.plot(bins[:-1], true_means, marker='o')
        plt.plot([-1, 1], [-1, 1], 'r--')
        plt.title('Value Prediction Calibration')
        plt.xlabel('Predicted Value')
        plt.ylabel('Average True Value')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

def evaluate_model(model_path: str, test_loader, device: str) -> Dict:
    """
    Convenience function to load a model and evaluate it.
    
    Args:
        model_path: Path to the saved model weights
        test_loader: DataLoader containing test data
        device: Device to run evaluation on
        
    Returns:
        Dictionary containing all computed metrics
    """
    # Load model
    model = PolicyValueNetwork().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Create evaluator and compute metrics
    evaluator = ModelEvaluator(model, device)
    metrics = evaluator.compute_metrics(test_loader)
    
    # Print summary
    print("\n=== Model Evaluation Summary ===")
    print(f"Policy Metrics:")
    print(f"  Start Position Accuracy: {metrics['policy_metrics']['start_accuracy']:.4f}")
    print(f"  End Position Accuracy: {metrics['policy_metrics']['end_accuracy']:.4f}")
    print(f"  Top-3 Move Accuracy: {metrics['policy_metrics']['top_k_accuracy'][3]:.4f}")
    print(f"\nValue Metrics:")
    print(f"  MSE: {metrics['value_metrics']['mse']:.4f}")
    print(f"  MAE: {metrics['value_metrics']['mae']:.4f}")
    print(f"  Correlation: {metrics['value_metrics']['correlation']:.4f}")
    
    # Create and save visualizations
    evaluator.plot_metrics(metrics, save_path="model_evaluation_plots.png")
    
    return metrics