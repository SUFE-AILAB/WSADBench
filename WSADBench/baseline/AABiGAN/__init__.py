"""
AA-BiGAN (Adversarial Autoencoder BiGAN) for Weakly Supervised Anomaly Detection

This module implements the AA-BiGAN algorithm for both tabular and computer vision data.

The implementation follows the paper:
"Adversarial Autoencoders for Weakly Supervised Anomaly Detection"

Main Components:
- AABiGAN: Main model class
- ModelFactory: Factory for creating models for different modalities
- Utility functions for training and evaluation

Usage:
    For tabular data:
        model = AABiGAN(modal='tabular', latent_dim=100, epochs=100)
        model.fit(X_train, y_train)
        scores = model.predict_proba(X_test)
    
    For CV data:
        model = AABiGAN(modal='cv', channels=3, img_size=32, epochs=200)
        model.fit(X_train, y_train)
        scores = model.predict_proba(X_test)
"""

from .run import AABiGAN
from .model import ModelFactory, AABiGANModel
from .fit import fit_aabigan, compute_anomaly_scores, create_auxiliary_data
from .utils import (
    weights_init, 
    compute_gradient_penalty,
    visualize_latent_space,
    visualize_reconstructions,
    compute_reconstruction_error_distribution,
    plot_loss_curves,
    compute_model_complexity,
    EarlyStopping
)

__all__ = [
    'AABiGAN',
    'ModelFactory', 
    'AABiGANModel',
    'fit_aabigan',
    'compute_anomaly_scores', 
    'create_auxiliary_data',
    'weights_init',
    'compute_gradient_penalty',
    'visualize_latent_space',
    'visualize_reconstructions', 
    'compute_reconstruction_error_distribution',
    'plot_loss_curves',
    'compute_model_complexity',
    'EarlyStopping'
]

__version__ = '1.0.0'
__author__ = 'WSADBench Team'
