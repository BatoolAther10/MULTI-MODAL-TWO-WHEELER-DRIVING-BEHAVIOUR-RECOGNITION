"""Factory for creating model adapters from the project registry."""

from __future__ import annotations

from typing import Dict, Type

from inference_layer.base_adapter import BaseOnDeviceModel
from inference_layer.stub_adapter import EgoDriveMaxAdapter, EgoDriveRTAdapter, StubAdapter
from inference_layer.dtaad_adapter import DTAADAdapter
from inference_layer.tabular_adapters import (
    HistGradientBoostingAdapter,
    LightGBMAdapter,
    LinearSVMAdapter,
    LogisticRegressionAdapter,
    MLPTabularAdapter,
    RBFSVMAdapter,
    RandomForestAdapter,
)

MODEL_REGISTRY: Dict[str, Type[BaseOnDeviceModel]] = {
    "stub": StubAdapter,
    "egodrive_rt": EgoDriveRTAdapter,
    "egodrive_max": EgoDriveMaxAdapter,
    "dtaad": DTAADAdapter,
    "logistic_regression": LogisticRegressionAdapter,
    "linear_svm": LinearSVMAdapter,
    "rbf_svm": RBFSVMAdapter,
    "random_forest": RandomForestAdapter,
    "histgradientboosting": HistGradientBoostingAdapter,
    "lightgbm": LightGBMAdapter,
    "mlp_tabular": MLPTabularAdapter,
}

try:
    from inference_layer.gru_adapter import GRUAdapter
    MODEL_REGISTRY["gru"] = GRUAdapter
except ImportError:
    pass

try:
    from inference_layer.lstm_adapter import LSTMAdapter
    MODEL_REGISTRY["lstm"] = LSTMAdapter
except ImportError:
    pass

try:
    from inference_layer.resnet1d_adapter import ResNet1DAdapter
    MODEL_REGISTRY["resnet1d"] = ResNet1DAdapter
except ImportError:
    pass

try:
    from inference_layer.convnet_adapter import ConvNetAdapter
    MODEL_REGISTRY["convnet_har"] = ConvNetAdapter
except ImportError:
    pass

try:
    from inference_layer.sgconv_adapter import SGConvAdapter
    MODEL_REGISTRY["sgconv"] = SGConvAdapter
except ImportError:
    pass

try:
    from inference_layer.s5_adapter import S5Adapter
    MODEL_REGISTRY["s5"] = S5Adapter
except ImportError:
    pass

try:
    from inference_layer.state_space_adapter import LRUAdapter, S4DAdapter, S5ONNXAdapter
    MODEL_REGISTRY["s4d"] = S4DAdapter
    MODEL_REGISTRY["s5_onnx"] = S5ONNXAdapter
    MODEL_REGISTRY["lru"] = LRUAdapter
except ImportError:
    pass

try:
    from inference_layer.tcnca_adapter import TCNCAAdapter
    MODEL_REGISTRY["tcnca"] = TCNCAAdapter
except ImportError:
    pass

try:
    from inference_layer.tst_adapter import TSTAdapter
    MODEL_REGISTRY["tst"] = TSTAdapter
except ImportError:
    pass

try:
    from inference_layer.transformer_adapter import TransformerAdapter
    for _name in ("tcnca_v1", "tcnca_v2", "tcnca_v3", "mega", "fusformer", "conv_transformer"):
        MODEL_REGISTRY[_name] = TransformerAdapter
except ImportError:
    pass

try:
    from inference_layer.xgboost_adapter import XGBoostAdapter
    MODEL_REGISTRY["xgboost"] = XGBoostAdapter
except ImportError:
    pass


def create_model(model_name: str, **kwargs: object) -> BaseOnDeviceModel:
    """Instantiate a registered model adapter by name."""
    try:
        model_cls = MODEL_REGISTRY[model_name.lower()]
    except KeyError as exc:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unsupported model '{model_name}'. Available: {available}") from exc
    return model_cls(**kwargs)
