"""BAGEL VAE/LLM flow connector module."""

from ... import OMNI_CONFIG_REGISTRY, OMNI_MODEL_REGISTRY


@OMNI_CONFIG_REGISTRY.register("bagel_flow_connector")
def register_bagel_flow_connector_config():
    from .configuration import BagelFlowConnectorConfig

    return BagelFlowConnectorConfig


@OMNI_MODEL_REGISTRY.register("bagel_flow_connector")
def register_bagel_flow_connector_model():
    from .modeling import BagelFlowConnector

    return BagelFlowConnector
