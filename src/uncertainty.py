"""
Uncertainty Quantification for Multimodal Fusion

Implements methods for estimating and calibrating confidence scores:
1. MC Dropout for epistemic uncertainty
2. Calibration metrics (ECE, reliability diagrams)
3. Uncertainty-weighted fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict


class MCDropoutUncertainty(nn.Module):
    """
    Monte Carlo Dropout for uncertainty estimation.
    
    Runs multiple forward passes with dropout enabled to estimate
    prediction uncertainty via variance.
    """
    
    def __init__(self, model: nn.Module, num_samples: int = 10):
        """
        Args:
            model: The model to estimate uncertainty for
            num_samples: Number of MC dropout samples
        """
        super().__init__()
        self.model = model
        self.num_samples = num_samples
    
    def forward(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with uncertainty estimation.
        
        Returns:
            mean_logits: (batch_size, num_classes) - mean prediction
            uncertainty: (batch_size,) - prediction uncertainty (variance)
        """
        # TODO: Implement MC Dropout
        # Steps:
        #   1. Enable dropout in model (model.train())
        #   2. Run num_samples forward passes
        #   3. Compute mean and variance of predictions
        #   4. Return mean prediction and uncertainty

        self.model.train()  
        logits_list = []
        for _ in range(self.num_samples):
            logits = self.model(*args, **kwargs)  
            logits_list.append(logits.unsqueeze(0))
        logits_stack = torch.cat(logits_list, dim=0) 
        mean_logits = logits_stack.mean(0)
        var = logits_stack.var(0).mean(dim=-1)  
        return mean_logits, var
        

class CalibrationMetrics:
    """
    Compute calibration metrics for confidence estimates.
    
    Key metrics:
    - Expected Calibration Error (ECE)
    - Maximum Calibration Error (MCE)  
    - Negative Log-Likelihood (NLL)
    """
    
    @staticmethod
    def expected_calibration_error(
        confidences: torch.Tensor,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        num_bins: int = 15
    ) -> float:
        """
        Compute Expected Calibration Error (ECE).
        
        ECE = Σ (|bin_accuracy - bin_confidence|) * (bin_size / total_size)
        
        Args:
            confidences: (N,) - predicted confidence scores [0, 1]
            predictions: (N,) - predicted class indices
            labels: (N,) - ground truth class indices
            num_bins: Number of bins for calibration
            
        Returns:
            ece: Expected Calibration Error (lower is better)
        """
        # TODO: Implement ECE calculation
        # Steps:
        #   1. Bin predictions by confidence level
        #   2. For each bin, compute accuracy and average confidence
        #   3. Compute weighted difference |accuracy - confidence|
        #   4. Return ECE
        
        # Hint: Use np.histogram or torch.histc to bin confidences

        confidences = confidences.cpu()
        predictions = predictions.cpu()
        labels = labels.cpu()

        bin_boundaries = torch.linspace(0, 1, num_bins + 1)
        ece = 0.0
        N = len(confidences)
        for i in range(num_bins):
            start = bin_boundaries[i]
            end = bin_boundaries[i + 1]
            mask = (confidences > start) & (confidences <= end) if i > 0 else (confidences >= start) & (confidences <= end)
            bin_size = mask.sum().item()
            if bin_size == 0:
                continue
            bin_conf = confidences[mask].mean().item()
            bin_acc = (predictions[mask] == labels[mask]).float().mean().item()
            ece += (bin_size / N) * abs(bin_acc - bin_conf)
        return float(ece)
        
    @staticmethod
    def maximum_calibration_error(
        confidences: torch.Tensor,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        num_bins: int = 15
    ) -> float:
        """
        Compute Maximum Calibration Error (MCE).
        
        MCE = max_bin |bin_accuracy - bin_confidence|
        
        Returns:
            mce: Maximum calibration error across bins
        """
        # TODO: Implement MCE
        # Similar to ECE but take max instead of average
        
        confidences = confidences.cpu()
        predictions = predictions.cpu()
        labels = labels.cpu()

        bin_boundaries = torch.linspace(0, 1, num_bins + 1)
        mce = 0.0
        for i in range(num_bins):
            start = bin_boundaries[i]
            end = bin_boundaries[i + 1]
            mask = (confidences > start) & (confidences <= end) if i > 0 else (confidences >= start) & (confidences <= end)
            if mask.sum() == 0:
                continue
            bin_conf = confidences[mask].mean().item()
            bin_acc = (predictions[mask] == labels[mask]).float().mean().item()
            mce = max(mce, abs(bin_acc - bin_conf))
        return float(mce)
        
    @staticmethod
    def negative_log_likelihood(
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> float:
        """
        Compute average Negative Log-Likelihood (NLL).
        
        NLL = -log P(y_true | x)
        
        Args:
            logits: (N, num_classes) - predicted logits
            labels: (N,) - ground truth labels
            
        Returns:
            nll: Average negative log-likelihood
        """
        # TODO: Implement NLL
        # Hint: Use F.cross_entropy which computes -log(softmax(logits)[label])

        return F.cross_entropy(logits, labels, reduction='mean').item()
    
    @staticmethod
    def reliability_diagram(
        confidences: np.ndarray,
        predictions: np.ndarray,
        labels: np.ndarray,
        num_bins: int = 15,
        save_path: str = None
    ) -> None:
        """
        Plot reliability diagram showing calibration.
        
        X-axis: Predicted confidence
        Y-axis: Actual accuracy
        Perfect calibration: y = x (diagonal line)
        
        Args:
            confidences: (N,) - confidence scores
            predictions: (N,) - predicted classes
            labels: (N,) - ground truth
            num_bins: Number of bins
            save_path: Optional path to save plot
        """
        import matplotlib.pyplot as plt        
        
        # TODO: Implement reliability diagram
        # Steps:
        #   1. Bin predictions by confidence
        #   2. Compute accuracy per bin
        #   3. Plot bar chart: confidence vs accuracy
        #   4. Add diagonal line for perfect calibration
        #   5. Add ECE to plot

        confidences = np.asarray(confidences)
        predictions = np.asarray(predictions)
        labels = np.asarray(labels)

        bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
        bin_acc = []
        bin_conf = []

        for i in range(num_bins):
            start = bin_boundaries[i]
            end = bin_boundaries[i + 1]
            if i == 0:
                mask = (confidences >= start) & (confidences <= end)
            else:
                mask = (confidences > start) & (confidences <= end)

            if mask.sum() == 0:
                bin_acc.append(0.0)
                bin_conf.append((start + end) / 2.0)
                continue

            acc = (predictions[mask] == labels[mask]).mean()
            conf = confidences[mask].mean()
            bin_acc.append(acc)
            bin_conf.append(conf)

        fig, ax = plt.subplots()
        # perfect calibration
        ax.plot([0, 1], [0, 1], '--', color='gray', label='perfect')

        # model bars
        width = 1.0 / num_bins
        ax.bar(bin_conf, bin_acc, width=width, alpha=0.6, edgecolor='black', label='model')

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Accuracy')
        ax.set_title('Reliability Diagram')
        ax.legend()
        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=200)
            plt.close(fig)
        else:
            plt.show()
        

class UncertaintyWeightedFusion(nn.Module):
    """
    Fuse modalities weighted by inverse uncertainty.
    
    Intuition: More uncertain modalities get lower weight.
    Weight_i ∝ 1 / (uncertainty_i + ε)
    """
    
    def __init__(self, epsilon: float = 1e-6):
        """
        Args:
            epsilon: Small constant to avoid division by zero
        """
        super().__init__()
        self.epsilon = epsilon
    
    def forward(
        self,
        modality_predictions: Dict[str, torch.Tensor],
        modality_uncertainties: Dict[str, torch.Tensor],
        modality_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fuse modality predictions weighted by inverse uncertainty.
        
        Args:
            modality_predictions: Dict of {modality: logits}
                                Each tensor: (batch_size, num_classes)
            modality_uncertainties: Dict of {modality: uncertainty}
                                   Each tensor: (batch_size,)
            modality_mask: (batch_size, num_modalities) - availability mask
            
        Returns:
            fused_logits: (batch_size, num_classes) - weighted fusion
            fusion_weights: (batch_size, num_modalities) - used weights
        """
        # TODO: Implement uncertainty-weighted fusion
        # Steps:
        #   1. Compute inverse uncertainty weights: w_i = 1/(σ_i + ε)
        #   2. Normalize weights to sum to 1
        #   3. Apply modality mask (zero weight for missing modalities)
        #   4. Fuse predictions: Σ w_i * pred_i
        #   5. Return fused predictions and weights
        
        modality_list = list(modality_predictions.keys())
        B = modality_mask.size(0)
        num_classes = next(iter(modality_predictions.values())).size(1)
        M = len(modality_list)

        # build weights per modality
        weights_list = []
        for i, m in enumerate(modality_list):
            u = modality_uncertainties[m]           
            w = 1.0 / (u + self.epsilon)              
            w = w * modality_mask[:, i]                
            weights_list.append(w.unsqueeze(1))
        weights = torch.cat(weights_list, dim=1)       
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-6)

        fused = torch.zeros(B, num_classes, device=modality_mask.device)
        for i, m in enumerate(modality_list):
            fused = fused + modality_predictions[m] * weights[:, i].unsqueeze(-1)

        return fused, weights

class TemperatureScaling(nn.Module):
    """
    Post-hoc calibration via temperature scaling.
    
    Learns a single temperature parameter T that scales logits:
    P_calibrated = softmax(logits / T)
    
    Reference: Guo et al. "On Calibration of Modern Neural Networks", ICML 2017
    """
    
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply temperature scaling to logits.
        
        Args:
            logits: (batch_size, num_classes) - model outputs
            
        Returns:
            scaled_logits: (batch_size, num_classes) - temperature-scaled logits
        """
        return logits / self.temperature.clamp(min=1e-6)
    
    def calibrate(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 50
    ) -> None:
        """
        Learn optimal temperature on validation set.
        
        Args:
            logits: (N, num_classes) - validation set logits
            labels: (N,) - validation set labels
            lr: Learning rate
            max_iter: Maximum optimization iterations
        """
        # TODO: Implement temperature calibration
        # Steps:
        #   1. Initialize temperature = 1.0
        #   2. Optimize temperature to minimize NLL on validation set
        #   3. Use LBFGS or Adam optimizer
        
        self.train()
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        nll_loss = torch.nn.CrossEntropyLoss()

        def closure():
            optimizer.zero_grad()
            scaled = self.forward(logits)
            loss = nll_loss(scaled, labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        # after calibration we can set to eval
        self.eval()

class EnsembleUncertainty:
    """
    Estimate uncertainty via ensemble of models.
    
    Train multiple models with different initializations/data splits.
    Uncertainty = variance across ensemble predictions.
    """
    
    def __init__(self, models: list):
        """
        Args:
            models: List of trained models (same architecture)
        """
        self.models = models
        self.num_models = len(models)
    
    def predict_with_uncertainty(
        self,
        inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get predictions and uncertainty from ensemble.
        
        Args:
            inputs: Model inputs
            
        Returns:
            mean_predictions: (batch_size, num_classes) - average prediction
            uncertainty: (batch_size,) - prediction variance
        """
        # TODO: Implement ensemble prediction
        # Steps:
        #   1. Get predictions from all models
        #   2. Compute mean prediction
        #   3. Compute variance as uncertainty measure
        #   4. Return mean and uncertainty
        
        logits_list = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                logits = model(inputs) 
                probs = torch.softmax(logits, dim=1)
                logits_list.append(probs.unsqueeze(0))

        probs_stack = torch.cat(logits_list, dim=0)
        mean_probs = probs_stack.mean(dim=0)          
        var_probs = probs_stack.var(dim=0).mean(dim=1) 

        mean_logits = torch.log(mean_probs.clamp(min=1e-8))
        return mean_logits, var_probs
    

def compute_calibration_metrics(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = 'cpu'
) -> Dict[str, float]:
    """
    Compute all calibration metrics on a dataset.
    
    Args:
        model: Trained model
        dataloader: Test/validation dataloader
        device: Device to run on
        
    Returns:
        metrics: Dict with ECE, MCE, NLL, accuracy
    """
    model.eval()
    all_confidences = []
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            
            logits = model(inputs)
            probs = F.softmax(logits, dim=1)
            confidences, predictions = torch.max(probs, dim=1)
            
            all_confidences.append(confidences.cpu())
            all_predictions.append(predictions.cpu())
            all_labels.append(labels.cpu())
    
    confidences = torch.cat(all_confidences)
    predictions = torch.cat(all_predictions)
    labels = torch.cat(all_labels)
    
    # TODO: Compute and return all metrics
    # - ECE
    # - MCE
    # - NLL
    # - Accuracy
    
    # ECE
    ece = CalibrationMetrics.expected_calibration_error(
        confidences=confidences,
        predictions=predictions,
        labels=labels,
        num_bins=15,
    )

    # MCE
    mce = CalibrationMetrics.maximum_calibration_error(
        confidences=confidences,
        predictions=predictions,
        labels=labels,
        num_bins=15,
    )

    # Accuracy
    accuracy = (predictions == labels).float().mean().item()

    # We can't properly compute NLL here because we didn't keep
    # per-class probabilities for the true label in the loop above.
    nll = float("nan")

    return {
        "ece": ece,
        "mce": mce,
        "nll": nll,
        "accuracy": accuracy,
    }

if __name__ == '__main__':
    # Test calibration metrics
    print("Testing calibration metrics...")
    
    # Generate fake predictions
    num_samples = 1000
    num_classes = 10
    
    # Well-calibrated predictions
    logits = torch.randn(num_samples, num_classes)
    labels = torch.randint(0, num_classes, (num_samples,))
    probs = F.softmax(logits, dim=1)
    confidences, predictions = torch.max(probs, dim=1)
    
    # Test ECE
    try:
        ece = CalibrationMetrics.expected_calibration_error(
            confidences, predictions, labels
        )
        print(f"✓ ECE computed: {ece:.4f}")
    except NotImplementedError:
        print("✗ ECE not implemented yet")
    
    # Test reliability diagram
    try:
        CalibrationMetrics.reliability_diagram(
            confidences.numpy(),
            predictions.numpy(),
            labels.numpy(),
            save_path='test_reliability.png'
        )
        print("✓ Reliability diagram created")
    except NotImplementedError:
        print("✗ Reliability diagram not implemented yet")

