"""
Shared types and helpers for the PegaFlow vLLM connector.
"""

import hashlib
import os
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata

from pegaflow.connector.connector_metrics import PegaKVConnectorStats, PegaPromMetrics
from pegaflow.logging_utils import get_connector_logger
from pegaflow.pegaflow import EngineRpcClient

if TYPE_CHECKING:
    from pegaflow.connector.state_manager import ServiceStateManager

logger = get_connector_logger()


@dataclass(frozen=True)
class ConnectorContext:
    """Shared configuration for scheduler/worker connectors."""

    instance_id: str
    namespace: str
    block_size: int
    num_layers: int
    tp_size: int
    world_size: int
    tp_rank: int | None
    device_id: int | None
    engine_client: EngineRpcClient
    state_manager: "ServiceStateManager"
    is_mla: bool = False

    @property
    def effective_tp_rank(self) -> int:
        """TP rank for PegaFlow server calls. MLA uses 0 since data is identical across ranks."""
        return 0 if self.is_mla else (self.tp_rank or 0)

    @property
    def effective_tp_size(self) -> int:
        """TP size for PegaFlow server calls. MLA uses 1 since only one copy is needed."""
        return 1 if self.is_mla else self.tp_size


@dataclass(frozen=True)
class LoadIntent:
    """Intent for a KV load operation."""

    block_ids: tuple[int, ...]
    block_hashes: tuple[bytes, ...]
    num_tokens: int


@dataclass(frozen=True)
class SaveIntent:
    """Intent for a KV save operation."""

    block_ids: tuple[int, ...]
    block_hashes: tuple[bytes, ...]


class PegaConnectorMetadata(KVConnectorMetadata):
    """Metadata passed from scheduler to worker for KV cache operations."""

    def __init__(
        self,
        load_intents: dict[str, LoadIntent] | None = None,
        save_intents: dict[str, SaveIntent] | None = None,
    ):
        super().__init__()
        # Maps request_id -> intent
        self.load_intents: dict[str, LoadIntent] = load_intents or {}
        self.save_intents: dict[str, SaveIntent] = save_intents or {}

    def __repr__(self) -> str:
        return (
            f"PegaConnectorMetadata(loads={len(self.load_intents)}, saves={len(self.save_intents)})"
        )


def parse_env_int(name: str, default: int) -> int:
    """Parse an integer from environment variable with fallback to default.

    Note: This function is typically called at module import time for class-level
    configuration. Changing the environment variable after module import will not
    affect values that were already read.

    Args:
        name: Environment variable name.
        default: Default value if env var is not set or invalid.

    Returns:
        Parsed integer value or default.
    """
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning("Invalid %s value '%s', using default %d", name, value, default)
        return default


def resolve_instance_id(vllm_config, dp_rank_suffix: bool = True) -> str:
    """Resolve or generate connector instance_id with optional DP rank suffix."""
    instance_id = vllm_config.kv_transfer_config.engine_id
    if instance_id:
        logger.info("[PegaKVConnector] Using kv_transfer_config.engine_id: %s", instance_id)
        return instance_id

    instance_id = vllm_config.instance_id or os.environ.get("PEGAFLOW_INSTANCE_ID", "")
    if not instance_id:
        instance_id = uuid.uuid4().hex
        logger.info(
            "[PegaKVConnector] No instance_id from vLLM; generated fallback %s",
            instance_id,
        )

    if dp_rank_suffix:
        parallel_config = vllm_config.parallel_config
        if parallel_config.data_parallel_size > 1:
            local_dp_rank = parallel_config.data_parallel_rank_local
            if local_dp_rank is not None:
                instance_id = f"{instance_id}_dp{local_dp_rank}"
                logger.info(
                    "[PegaKVConnector] Appended DP rank to instance_id: %s (dp_size=%d, local_dp_rank=%d)",
                    instance_id,
                    parallel_config.data_parallel_size,
                    local_dp_rank,
                )

    return instance_id


def derive_namespace(vllm_config, tp_size: int) -> str:
    """
    Derive namespace for storage isolation.
    """
    model_config = vllm_config.model_config
    cache_config = vllm_config.cache_config

    factors = {
        "model": model_config.model,
        "dtype": str(model_config.dtype),
        "tp_size": tp_size,
        "num_kv_heads": model_config.get_total_num_kv_heads(),
        "head_size": model_config.get_head_size(),
        "num_hidden_layers": model_config.get_total_num_hidden_layers(),
        "cache_dtype": str(cache_config.cache_dtype),
    }

    factor_str = str(sorted(factors.items()))
    hash_suffix = hashlib.sha256(factor_str.encode()).hexdigest()[:8]
    return f"{hash_suffix}"


def detect_mla(vllm_config) -> bool:
    """Detect if the model uses Multi-head Latent Attention (e.g. DeepSeek V2/V3)."""
    hf_config = vllm_config.model_config.hf_text_config
    return getattr(hf_config, "kv_lora_rank", None) is not None


def is_indexer_layer(layer_name: str) -> bool:
    """
    Check if a layer is an indexer layer used for sparse attention.

    Indexer layers (DeepseekV32IndexerCache) are used for sparse attention's top-k
    token selection. They have special CUDA Graph constraints and should be excluded
    from KV transfer to avoid CPU-GPU synchronization issues during graph capture.

    Args:
        layer_name: The attention layer name (e.g., "model.layers.0.self_attn.indexer.attn")

    Returns:
        True if the layer is an indexer layer.
    """
    return "indexer" in layer_name


def is_draft_layer(layer_name: str, vllm_config) -> bool:
    """
    Check if a layer is a draft/MTP layer that should be excluded from KV transfer.

    MTP (Multi-Token Prediction) layers are additional layers beyond the target model's
    hidden layers, used for speculative decoding. Their KV cache contains draft tokens
    that may be rejected, so they should not be saved to or loaded from external storage.

    Args:
        layer_name: The attention layer name (e.g., "model.layers.78.self_attn.attn")
        vllm_config: The vLLM configuration

    Returns:
        True if the layer is a MTP layer that should be excluded.
    """
    spec_config = vllm_config.speculative_config
    if spec_config is None:
        return False

    # Check for MTP block pattern (some models use explicit mtp_block naming)
    if "mtp_block" in layer_name:
        return True

    # Check layer index: MTP layers have index >= num_hidden_layers
    # Layer names follow pattern: "model.layers.{idx}.self_attn.attn"
    try:
        parts = layer_name.split(".")
        for i, part in enumerate(parts):
            if part == "layers" and i + 1 < len(parts):
                layer_idx = int(parts[i + 1])
                num_hidden_layers = vllm_config.model_config.get_num_layers(
                    vllm_config.parallel_config
                )
                if layer_idx >= num_hidden_layers:
                    return True
                break
    except (ValueError, IndexError):
        pass

    return False


def should_exclude_from_transfer(layer_name: str, vllm_config) -> bool:
    """
    Check if a layer should be excluded from KV transfer.

    Layers are excluded if they are:
    1. Indexer layers (sparse attention's top-k selection, has CUDA Graph constraints)
    2. MTP/draft layers (speculative decoding, KV contains potentially rejected tokens)

    Args:
        layer_name: The attention layer name
        vllm_config: The vLLM configuration

    Returns:
        True if the layer should be excluded from KV transfer.
    """
    # Always exclude indexer layers (CUDA Graph compatibility)
    if is_indexer_layer(layer_name):
        return True

    # Exclude MTP layers when speculative decoding is enabled
    if is_draft_layer(layer_name, vllm_config):
        return True

    return False


__all__ = [
    "ConnectorContext",
    "LoadIntent",
    "PegaConnectorMetadata",
    "PegaKVConnectorStats",
    "PegaPromMetrics",
    "SaveIntent",
    "derive_namespace",
    "detect_mla",
    "is_draft_layer",
    "is_indexer_layer",
    "logger",
    "parse_env_int",
    "resolve_instance_id",
    "should_exclude_from_transfer",
]
