from .base import BaseMetric
from .base_online import OnlineMetric
from .ridge import RidgeMetric, TorchRidgeMetric, Ridge3DChunkedMetric, InverseRidgeChunkedMetric
from .ridge import RidgeAutoMetric, TorchLassoMetric, TorchElasticMetric
from .pls import PLSMetric
from .bidirectional import BidirectionalMappingMetric
from .one_to_one import OneToOneMappingMetric
from .soft_matching import SoftMatchingMetric
from .semi_matching import SemiMatchingMetric
from .rsa import RSAMetric, TemporalRSAMetric
from .versa import VeRSAMetric
from .online_mappers import OnlineLinearClassifier, OnlineLinearRegressor
from .online_mappers import OnlineTransformerClassifier
from .orientation_selectivity import OrientationSelectivity
from .physion import OnlinePhysionContactDetection, OnlinePhysionContactPrediction, OnlinePhysionPlacementDetection, OnlinePhysionPlacementPrediction


METRICS = {
    "ridge": RidgeMetric,
    "torch_ridge": TorchRidgeMetric,
    "torch_lasso": TorchLassoMetric,
    "torch_elastic": TorchElasticMetric,
    "pls": PLSMetric,
    "bidirectional": BidirectionalMappingMetric,
    "one_to_one": OneToOneMappingMetric,
    "soft_matching": SoftMatchingMetric,
    "semi_matching": SemiMatchingMetric,
    "rsa": RSAMetric,
    "temporal_rsa": TemporalRSAMetric,
    "versa": VeRSAMetric,
    "temporal_ridge": Ridge3DChunkedMetric,
    "inverse_ridge": InverseRidgeChunkedMetric,

    # online metrics
    "online_linear_classifier": OnlineLinearClassifier,
    "online_linear_regressor": OnlineLinearRegressor,
    "online_transformer_classifier": OnlineTransformerClassifier,
    "physion_placement_prediction": OnlinePhysionPlacementPrediction,
    "physion_placement_detection": OnlinePhysionPlacementDetection,
    "physion_contact_prediction": OnlinePhysionContactPrediction,
    "physion_contact_detection": OnlinePhysionContactDetection,
    # topographic metrics
    "orientation_selectivity": OrientationSelectivity,
}


__all__ = ["BaseMetric", "OnlineMetric", "METRICS"]  # Add OnlineMetric
