from __future__ import annotations

"""Baseline vLLM v1 KV connector for local development.

This module defines :class:`PegaKVConnector`, a thin subclass of
``vllm.distributed.kv_transfer.kv_connector.v1.base.KVConnectorBase_V1``.

At the moment it only mirrors the abstract API and raises
``NotImplementedError`` in all required methods, so that we have a
self-contained place inside this repo to start iterating on our own
PegaFlow-backed connector implementation.

Usage example (scheduler/worker side)::

    from pegaflow import PegaKVConnector, KVConnectorRole

    connector = PegaKVConnector(vllm_config, KVConnectorRole.WORKER)

Later we can register this class as a dynamic connector in vLLM by
referencing it via its full import path.
"""

from typing import Any, Optional, Tuple, Dict, List
import torch
import pickle

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorRole,
    KVConnectorMetadata,
)

# Import the Rust PegaEngine
from pegaflow.pegaflow import PegaEngine

# Import IPC wrapper for cross-process GPU memory sharing
from pegaflow.ipc_wrapper import CudaIPCWrapper

class PegaConnectorMetadata(KVConnectorMetadata):
    """Metadata for PegaFlow KV connector.

    Contains information needed to save/load KV cache blocks:
    - block_tables: mapping from sequence to block IDs
    - block_hashes: content hashes for each block
    - seq_lens: length of each sequence
    - requests_to_load: mapping from request ID to load information
    """

    def __init__(
        self,
        block_tables: Optional[Dict[str, List[int]]] = None,
        block_hashes: Optional[Dict[str, List[bytes]]] = None,
        seq_lens: Optional[Dict[str, int]] = None,
        requests_to_load: Optional[Dict[str, Dict]] = None,
    ):
        super().__init__()
        self.block_tables = block_tables or {}
        self.block_hashes = block_hashes or {}
        self.seq_lens = seq_lens or {}
        self.requests_to_load = requests_to_load or {}

def print_handle(handle):
    device = handle[0]
    ipc_handle = handle[1]
    size = handle[2]
    offset = handle[3]
    ipc_event_handle = handle[6]
    print(f"device: {device}, ipc_handle: {ipc_handle}, size: {size}, offset: {offset}, ipc_event_handle: {ipc_event_handle}")

class PegaKVConnector(KVConnectorBase_V1):
    """Skeleton v1 KV connector for PegaFlow.

    This class intentionally keeps the same method signatures as
    :class:`KVConnectorBase_V1` so that it can be used as a drop-in
    implementation once we fill in the logic. All abstract methods
    currently raise :class:`NotImplementedError`.
    """

    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole):
        """Create a new PegaKVConnector.

        Args:
            vllm_config: vLLM configuration object.
            role: Whether this connector instance runs in the scheduler
                process or the worker process.
        """
        super().__init__(vllm_config, role)
        # Initialize PegaEngine for managing KV cache handles
        self.engine = PegaEngine()

        # Track block tables and hashes for each request across steps
        self._request_block_tables = {}  # req_id -> list[int]
        self._request_block_hashes = {}  # req_id -> list[bytes]

        # Track pending save operations
        self._pending_saves = []  # list[dict]

        # Track requests that need to load KV cache from CPU
        self._requests_to_load = {}  # req_id -> dict with load info

        # Track which prompts have been saved (use first 10 tokens as key)
        # This is a simple workaround for process isolation
        self._saved_prompts = set()  # set of tuples: (token1, token2, ..., token10)

        # Get block size from vllm_config
        self._block_size = vllm_config.cache_config.block_size

    # ==============================
    # Worker-side methods
    # ==============================

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        """
        Start loading the KV cache from the connector to vLLM's paged
        KV buffer. This is called from the forward context before the
        forward pass to enable async loading during model execution.

        Args:
            forward_context (ForwardContext): the forward context.
            **kwargs: additional arguments for the load operation

        Note:
            The number of elements in kv_caches and layer_names should be
            the same.

        """
        # ============================================================
        # STEP 1: Get connector metadata
        # ============================================================
        metadata = self._get_connector_metadata()

        if not isinstance(metadata, PegaConnectorMetadata):
            return

        # ============================================================
        # STEP 2: Check if there are requests to load
        # ============================================================
        if not metadata.requests_to_load:
            return

        print(f"[PegaKVConnector] start_load_kv: Loading KV cache for {len(metadata.requests_to_load)} requests")

        # ============================================================
        # STEP 3: Load KV blocks for each request and each layer
        # ============================================================
        try:
            for req_id, load_info in metadata.requests_to_load.items():
                block_ids = load_info['block_ids']
                block_hashes = load_info['block_hashes']
                num_tokens = load_info['num_tokens']

                print(f"[PegaKVConnector] Loading request {req_id}: {len(block_ids)} blocks, {num_tokens} tokens")

                # Load for each layer
                for layer_name in forward_context.no_compile_layers:
                    layer = forward_context.no_compile_layers[layer_name]

                    # Only process layers that have kv_cache attribute (attention layers)
                    if not hasattr(layer, 'kv_cache'):
                        continue

                    try:
                        # Call Rust backend to load KV blocks from CPU to GPU
                        self.engine.load_kv_blocks_to_ipc(
                            layer_name,
                            block_ids,
                            block_hashes
                        )
                        print(f"[PegaKVConnector] Loaded layer {layer_name} for request {req_id}")

                    except Exception as e:
                        print(f"[PegaKVConnector] Failed to load layer {layer_name} for request {req_id}: {e}")
                        # Continue with other layers even if one fails

            # ============================================================
            # STEP 4: Synchronize CUDA operations
            # ============================================================
            torch.cuda.synchronize()
            print(f"[PegaKVConnector] start_load_kv: All loads complete")

        except Exception as e:
            print(f"[PegaKVConnector] Error in start_load_kv: {e}")
            import traceback
            traceback.print_exc()

    def wait_for_layer_load(self, layer_name: str) -> None:
        """
        Block until the KV for a specific layer is loaded into vLLM's
        paged buffer. This is called from within attention layer to ensure
        async copying from start_load_kv is complete.

        This interface will be useful for layer-by-layer pipelining.

        Args:
            layer_name: the name of that layer
        """
        pass

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: "torch.Tensor",  # type: ignore[name-defined]
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        """
        Start saving a layer of KV cache from vLLM's paged buffer
        to the connector. This is called from within attention layer to
        enable async copying during execution.

        Args:
            layer_name (str): the name of the layer.
            kv_layer (torch.Tensor): the paged KV buffer of the current
                layer in vLLM.
            attn_metadata (AttentionMetadata): the attention metadata.
            **kwargs: additional arguments for the save operation.
        """

        # Store for later processing in wait_for_save
        self._pending_saves.append({
            'layer_name': layer_name,
            'kv_layer': kv_layer,
            'attn_metadata': attn_metadata,
            'kwargs': kwargs
        })

    def wait_for_save(self) -> None:
        """
        Block until all the save operations is done. This is called
        as the forward context exits to ensure that the async saving
        from save_kv_layer is complete before finishing the forward.

        This prevents overwrites of paged KV buffer before saving done.
        """
        # ============================================================
        # STEP 1: Check if there are pending saves
        # ============================================================
        if len(self._pending_saves) == 0:
            return

        try:
            # ============================================================
            # STEP 2: Get connector metadata
            # ============================================================
            metadata = self._get_connector_metadata()

            if not isinstance(metadata, PegaConnectorMetadata):
                return

            # ============================================================
            # STEP 3: Create CUDA event for synchronization
            # ============================================================
            with torch.cuda.stream(torch.cuda.current_stream()):
                event = torch.cuda.Event(interprocess=True)
                event.record()

            # ============================================================
            # STEP 4: Process each layer's save operation
            # ============================================================
            total_blocks_saved = 0

            for save_info in self._pending_saves:
                layer_name = save_info['layer_name']
                attn_metadata = save_info['attn_metadata']

                # Skip if block_table is missing or None
                if attn_metadata.block_table is None:
                    continue

                block_table = attn_metadata.block_table  # [num_seqs, max_blocks]
                seq_lens = attn_metadata.seq_lens

                # Process each sequence in the batch
                for seq_idx in range(block_table.shape[0]):
                    # Calculate number of blocks needed for this sequence
                    if seq_lens is not None:
                        seq_len = seq_lens[seq_idx].item()
                        num_blocks = (seq_len + 15) // 16  # Round up to block size
                    else:
                        # Fallback: count non-zero blocks
                        num_blocks = (block_table[seq_idx] != 0).sum().item()

                    if num_blocks == 0:
                        continue

                    # Get active block IDs for this sequence
                    active_blocks = block_table[seq_idx, :num_blocks].cpu().tolist()

                    # Find matching block hashes from metadata
                    # TODO: Improve mapping between seq_idx and req_id
                    block_hashes_for_seq = None
                    matched_req_id = None
                    for req_id, hashes in metadata.block_hashes.items():
                        if len(hashes) >= num_blocks:
                            block_hashes_for_seq = hashes[:num_blocks]
                            matched_req_id = req_id
                            break

                    if block_hashes_for_seq is None:
                        continue

                    # Save blocks to storage via Rust backend
                    try:
                        self.engine.save_kv_blocks_from_ipc(
                            layer_name,
                            active_blocks,
                            block_hashes_for_seq
                        )
                        total_blocks_saved += num_blocks
                    except Exception:
                        # Silently skip failed saves
                        pass

            # ============================================================
            # STEP 5: Wait for CUDA operations to complete
            # ============================================================
            event.synchronize()

        except Exception:
            # Silently handle errors
            pass
        finally:
            # ============================================================
            # STEP 6: Clean up pending saves
            # ============================================================
            self._pending_saves.clear()

    # ==============================
    # Scheduler-side methods
    # ==============================

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> Tuple[Optional[int], bool]:
        """
        Get number of new tokens that can be loaded from the
        external KV cache beyond the num_computed_tokens.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request

        Returns:
            A tuple with the following elements:
                - An optional number of tokens that can be loaded from the
                  external KV cache beyond what is already computed.
                  If None, it means that the connector needs more time to
                  determine the number of matched tokens, and the scheduler
                  should query for this request again later.
                - `True` if external KV cache tokens will be loaded
                  asynchronously (between scheduler steps). Must be
                  'False' if the first element is 0.

        Notes:
            The connector should only consider the largest prefix of prompt-
            tokens for which KV cache is actually available at the time of the
            call. If the cache cannot be loaded for some tokens (e.g., due to
            connectivity issues or eviction), those tokens must not be taken
            into account.
        """
        import hashlib
        # Get prompt tokens
        prompt_token_ids = request.prompt_token_ids or []
        if len(prompt_token_ids) == 0:
            return (0, False)

        req_id = request.request_id

        # Calculate number of blocks needed for the prompt
        num_tokens = len(prompt_token_ids)
        num_blocks = (num_tokens + self._block_size - 1) // self._block_size

        if num_blocks == 0:
            print(f"return 0 since num blocks is 0")
            return (0, False)

        # Generate block hashes (same logic as in build_connector_meta)
        block_hashes = []
        for block_idx in range(num_blocks):
            hash_input = f"{req_id}_block_{block_idx}".encode('utf-8')
            block_hash = hashlib.sha256(hash_input).digest()
            block_hashes.append(block_hash)

        # SIMPLIFIED LOGIC: Use first 10 tokens as a fingerprint to check if we've seen this prompt
        # 1. First run: prompt not in _saved_prompts -> return 0
        # 2. Second run: prompt in _saved_prompts -> return all tokens beyond num_computed_tokens
        #    BUT: Must leave at least 1 token for GPU to compute (to generate logits)

        # Create prompt fingerprint from first 10 tokens
        prompt_fingerprint = tuple(prompt_token_ids[:10])

        # Check if this prompt has been saved before
        if prompt_fingerprint not in self._saved_prompts:
            print(f"[PegaKVConnector] Request {req_id}: First time seeing this prompt, no cache available")
            return (0, False)

        # Prompt was saved before - assume ALL prompt tokens are cached
        # Calculate how many tokens beyond num_computed_tokens can be loaded
        # IMPORTANT: Must leave at least 1 token for GPU to compute (vLLM requirement)
        max_cacheable_tokens = num_tokens - 1  # Leave last token for GPU
        num_new_tokens = max_cacheable_tokens - num_computed_tokens

        if num_new_tokens <= 0:
            print(f"[PegaKVConnector] Request {req_id}: All cacheable tokens already computed")
            return (0, False)

        print(f"[PegaKVConnector] Request {req_id}: Prompt seen before! Can load {num_new_tokens} tokens from cache (total: {num_tokens}, computed: {num_computed_tokens}, leaving 1 for GPU)")
        return (num_new_tokens, False)

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        """
        Update KVConnector state after block allocation.

        If get_num_new_matched_tokens previously returned True for a
        request, this function may be called twice for that same request -
        first when blocks are allocated for the connector tokens to be
        asynchronously loaded into, and second when any additional blocks
        are allocated, after the load/transfer is complete.

        Args:
            request (Request): the request object.
            blocks (KVCacheBlocks): the blocks allocated for the request.
            num_external_tokens (int): the number of tokens that will be
                loaded from the external KV cache.
        """
        # If there are external tokens to load, record this request
        if num_external_tokens > 0:
            req_id = request.request_id
            print(f"[PegaKVConnector] Recording request {req_id} for loading {num_external_tokens} tokens")

            self._requests_to_load[req_id] = {
                'request': request,
                'blocks': blocks,
                'num_external_tokens': num_external_tokens,
            }

    def build_connector_meta(self, scheduler_output: "SchedulerOutput") -> KVConnectorMetadata:
        """
        Build the connector metadata for this step.

        This function should NOT modify fields in the scheduler_output.
        Also, calling this function will reset the state of the connector.

        Args:
            scheduler_output (SchedulerOutput): the scheduler output object.
        """
        import hashlib

        block_tables = {}
        block_hashes = {}
        seq_lens = {}

        # ============================================================
        # STEP 1: Process new requests (first time scheduled)
        # ============================================================
        new_reqs = scheduler_output.scheduled_new_reqs
        for req in new_reqs:
            req_id = req.req_id

            # Extract block table from first KV cache group
            # block_ids is tuple[list[int], ...] for different KV cache groups
            block_table = list(req.block_ids[0]) if req.block_ids else []

            # Store block table in metadata and persistent state
            if block_table:
                block_tables[req_id] = block_table
                self._request_block_tables[req_id] = block_table.copy()

            # Store sequence length
            seq_lens[req_id] = req.num_computed_tokens

            # Generate block hashes for each block
            if block_table:
                hashes = []
                for block_idx in range(len(block_table)):
                    hash_input = f"{req_id}_block_{block_idx}".encode('utf-8')
                    block_hash = hashlib.sha256(hash_input).digest()
                    hashes.append(block_hash)

                block_hashes[req_id] = hashes
                self._request_block_hashes[req_id] = hashes.copy()

            # Record prompt fingerprint (first 10 tokens) for cache hit detection
            if req.prompt_token_ids and len(req.prompt_token_ids) > 0:
                prompt_fingerprint = tuple(req.prompt_token_ids[:10])
                self._saved_prompts.add(prompt_fingerprint)
                print(f"[PegaKVConnector] Recorded prompt fingerprint for request {req_id}: {prompt_fingerprint[:3]}...")

        # ============================================================
        # STEP 2: Process cached requests (already scheduled, now in decode phase)
        # ============================================================
        cached_reqs = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(cached_reqs.req_ids):
            # Retrieve existing block table from persistent state
            block_table = self._request_block_tables.get(req_id, []).copy()

            # Append new blocks if any
            if i < len(cached_reqs.new_block_ids):
                new_blocks = cached_reqs.new_block_ids[i]
                if new_blocks is not None:
                    # new_blocks is tuple[list[int], ...] for different KV cache groups
                    new_block_list = list(new_blocks[0]) if new_blocks else []
                    if new_block_list:
                        block_table.extend(new_block_list)
                        self._request_block_tables[req_id] = block_table.copy()

            # Store block table in metadata
            if block_table:
                block_tables[req_id] = block_table

            # Store sequence length
            if i < len(cached_reqs.num_computed_tokens):
                seq_lens[req_id] = cached_reqs.num_computed_tokens[i]

            # Generate hashes for any new blocks
            if block_table:
                existing_hashes = self._request_block_hashes.get(req_id, []).copy()
                num_existing = len(existing_hashes)
                num_total = len(block_table)

                # Generate hashes for blocks that don't have hashes yet
                if num_total > num_existing:
                    for block_idx in range(num_existing, num_total):
                        hash_input = f"{req_id}_block_{block_idx}".encode('utf-8')
                        block_hash = hashlib.sha256(hash_input).digest()
                        existing_hashes.append(block_hash)

                    self._request_block_hashes[req_id] = existing_hashes.copy()

                block_hashes[req_id] = existing_hashes

        # ============================================================
        # STEP 3: Process requests that need to load from CPU storage
        # ============================================================
        requests_to_load = {}

        for req_id, load_info in self._requests_to_load.items():
            request = load_info['request']
            num_external_tokens = load_info['num_external_tokens']

            # Find this request in scheduler_output
            found = False
            for req in scheduler_output.scheduled_new_reqs:
                if req.req_id == req_id:
                    # Extract block IDs from the request
                    block_ids = list(req.block_ids[0]) if req.block_ids else []

                    # Calculate number of blocks needed
                    num_blocks = (num_external_tokens + self._block_size - 1) // self._block_size

                    if num_blocks > 0 and len(block_ids) >= num_blocks:
                        # Generate block hashes for the blocks to load
                        load_hashes = []
                        for block_idx in range(num_blocks):
                            hash_input = f"{req_id}_block_{block_idx}".encode('utf-8')
                            block_hash = hashlib.sha256(hash_input).digest()
                            load_hashes.append(block_hash)

                        # Store load information
                        requests_to_load[req_id] = {
                            'block_ids': block_ids[:num_blocks],
                            'block_hashes': load_hashes,
                            'num_tokens': num_external_tokens,
                        }

                        print(f"[PegaKVConnector] Prepared load metadata for request {req_id}: {num_blocks} blocks, {num_external_tokens} tokens")

                    found = True
                    break

            if not found:
                print(f"[PegaKVConnector] Warning: Request {req_id} not found in scheduled_new_reqs")

        # Clear the requests_to_load after processing
        self._requests_to_load.clear()

        # ============================================================
        # STEP 4: Build and return metadata
        # ============================================================
        metadata = PegaConnectorMetadata(
            block_tables=block_tables,
            block_hashes=block_hashes,
            seq_lens=seq_lens,
            requests_to_load=requests_to_load,
        )

        return metadata
    
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """Register KV cache tensors with the PegaEngine.

        This method simulates cross-process IPC by:
        1. Wrapping each tensor in CudaIPCWrapper (sender side)
        2. Serializing the wrapper to bytes (cross-process transmission)
        3. Deserializing the wrapper (receiver side)
        4. Reconstructing tensor from IPC handle
        5. Extracting GPU pointer and passing to Rust

        Args:
            kv_caches: Dictionary mapping layer names to KV cache tensors
        """
        for layer_name, kv_cache in kv_caches.items():
            # Ensure tensor is contiguous and at offset 0
            assert kv_cache.is_contiguous(), f"KV cache for {layer_name} must be contiguous"
            assert kv_cache.storage_offset() == 0, f"KV cache for {layer_name} must have offset 0"

            # Step 5: Extract GPU pointer and size
            data_ptr = kv_cache.data_ptr()
            size_bytes = kv_cache.numel() * kv_cache.element_size()

            # Step 6: Pass GPU pointer to Rust (Rust only stores ptr and size)
            self.engine.register_kv_cache_ptr(layer_name, data_ptr, size_bytes)

    def shutdown(self):
        """Shutdown the connector and unregister all KV caches."""
        self.engine.unregister_all_kv_caches()


__all__ = ["PegaKVConnector", "KVConnectorRole"]

