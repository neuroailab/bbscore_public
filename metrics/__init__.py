from .base import BaseMetric
from .base_online import OnlineMetric
from .behavioral_classifier import BehavioralRegressionMetric, TorchBehavioralRegressionMetric
from .ridge import RidgeMetric, TorchRidgeMetric, RidgeChunkedMetric, Ridge3DChunkedMetric, InverseRidgeChunkedMetric
from .ridge import RidgeAutoMetric, TorchLassoMetric, TorchElasticMetric, TorchWoodburyMetric, TorchBlockMetric
from .macaque import RidgeExperimentsMetric
from .pls import PLSMetric
from .bidirectional import BidirectionalMappingMetric
from .one_to_one import OneToOneMappingMetric
from .soft_matching import SoftMatchingMetric
from .semi_matching import SemiMatchingMetric
from .sgd_regressor import SGDMetric
from .rsa import RSAMetric, TemporalRSAMetric, RepetitionRSAMetric, DynamicRSAMetric, TemporalRepetitionRSAMetric
from .versa import VeRSAMetric
from .online_mappers import OnlineLinearClassifier, OnlineLinearRegressor
from .online_mappers import OnlineTransformerClassifier, OnlineTransformerRegressor, OnlineNeuralTransformerRegressor
from .online_mappers import OnlineAttentionRegressor
from .orientation_selectivity import OrientationSelectivity
from .physion import OnlinePhysionContactDetection, OnlinePhysionContactPrediction, OnlinePhysionPlacementDetection, OnlinePhysionPlacementPrediction


METRICS = {
    "ridge": RidgeMetric,
    "torch_ridge": TorchRidgeMetric,
    "torch_lasso": TorchLassoMetric,
    "torch_elastic": TorchElasticMetric,
    "torch_woodbury_ridge": TorchWoodburyMetric,
    "torch_block_ridge": TorchBlockMetric,
    "torch_diagonal_ridge": RidgeExperimentsMetric,
    "chunked_ridge": RidgeChunkedMetric,
    "ridge_auto": RidgeAutoMetric,
    "pls": PLSMetric,
    "bidirectional": BidirectionalMappingMetric,
    "one_to_one": OneToOneMappingMetric,
    "soft_matching": SoftMatchingMetric,
    "semi_matching": SemiMatchingMetric,
    "rsa": RSAMetric,
    "temporal_rsa": TemporalRSAMetric,
    "repetition_rsa": RepetitionRSAMetric,
    "dynamic_rsa": DynamicRSAMetric,
    "temporal_repetition_rsa": TemporalRepetitionRSAMetric,
    "versa": VeRSAMetric,
    "behavioral_regression": BehavioralRegressionMetric,
    "torch_behavioral_regression": TorchBehavioralRegressionMetric,
    "temporal_ridge": Ridge3DChunkedMetric,
    "inverse_ridge": InverseRidgeChunkedMetric,
    "sgd_regressor": SGDMetric,

    # online metrics
    "online_linear_classifier": OnlineLinearClassifier,
    "online_linear_regressor": OnlineLinearRegressor,
    "online_transformer_classifier": OnlineTransformerClassifier,
    "online_transformer_regressor": OnlineTransformerRegressor,
    "online_neural_transformer": OnlineNeuralTransformerRegressor,
    "online_attention_regressor": OnlineAttentionRegressor,
    "physion_placement_prediction": OnlinePhysionPlacementPrediction,
    "physion_placement_detection": OnlinePhysionPlacementDetection,
    "physion_contact_prediction": OnlinePhysionContactPrediction,
    "physion_contact_detection": OnlinePhysionContactDetection,
    # topographic metrics
    "orientation_selectivity": OrientationSelectivity,
}


__all__ = ["BaseMetric", "OnlineMetric", "METRICS"]  # Add OnlineMetric
