"""Cosmos3 flow connector: latent <-> generation-tower rows, and the diffusion loss."""

from ... import OMNI_ACCELERATED_MODEL_REGISTRY, OMNI_CONFIG_REGISTRY, OMNI_MODEL_REGISTRY


@OMNI_CONFIG_REGISTRY.register("cosmos3_flow_connector")
def register_cosmos3_flow_connector_config():
    from .configuration import Cosmos3FlowConnectorConfig

    return Cosmos3FlowConnectorConfig


@OMNI_MODEL_REGISTRY.register("cosmos3_flow_connector")
def register_cosmos3_flow_connector_model():
    from .modeling import Cosmos3FlowConnector

    return Cosmos3FlowConnector


@OMNI_ACCELERATED_MODEL_REGISTRY.register("cosmos3_flow_connector")
def register_cosmos3_flow_connector_accelerated_model():
    from .accelerated import Cosmos3FlowConnectorAccelerated

    return Cosmos3FlowConnectorAccelerated
