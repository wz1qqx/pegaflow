"""CUDA IPC Wrapper for cross-process GPU memory sharing.

This module provides a wrapper class for PyTorch tensors that enables
cross-process GPU memory sharing via CUDA IPC handles. The wrapper can
be serialized (via pickle) and sent across process boundaries.
"""

import torch
from typing import Tuple


class CudaIPCWrapper:
    """Wrapper for CUDA IPC handle with tensor metadata.
    
    This class wraps a PyTorch CUDA tensor and extracts its IPC handle,
    allowing the tensor to be reconstructed in another process.
    
    Attributes:
        handle: CUDA IPC handle tuple (device, ipc_handle, size, offset, ...)
        dtype: PyTorch dtype of the tensor
        shape: Shape tuple of the tensor
        device: CUDA device index
    
    Example:
        # Process 1 (sender)
        tensor = torch.randn(10, device='cuda:0')
        wrapper = CudaIPCWrapper(tensor)
        serialized = pickle.dumps(wrapper)
        # ... send serialized bytes to another process ...
        
        # Process 2 (receiver)
        wrapper = pickle.loads(serialized)
        tensor = wrapper.to_tensor()  # Reconstruct tensor
        ptr = tensor.data_ptr()  # Get GPU pointer
    """
    
    def __init__(self, tensor: torch.Tensor):
        """Create IPC wrapper from a CUDA tensor.
        
        Args:
            tensor: PyTorch CUDA tensor to wrap. Must be contiguous and
                   have zero storage offset.
        
        Raises:
            AssertionError: If tensor is not contiguous or has non-zero offset.
        """
        assert tensor.storage_offset() == 0, "Tensor must have zero storage offset"
        assert tensor.is_contiguous(), "Tensor must be contiguous"
        
        # Get the underlying storage and create IPC handle
        storage = tensor.untyped_storage()
        handle = storage._share_cuda_()
        
        # Store metadata needed to reconstruct the tensor
        self.handle = handle
        self.dtype = tensor.dtype
        self.shape = tensor.shape
        self.device = tensor.device.index
    
    def to_tensor(self) -> torch.Tensor:
        """Reconstruct tensor from IPC handle.
        
        This method creates a new tensor in the current process that shares
        the same GPU memory as the original tensor (via CUDA IPC).
        
        Returns:
            PyTorch tensor that shares GPU memory with the original tensor.
        
        Note:
            The reconstructed tensor shares memory with the original. Any
            modifications to one will be visible in the other.
        """
        # Reconstruct storage from IPC handle
        storage = torch.UntypedStorage._new_shared_cuda(*self.handle)
        
        # Get device from handle
        device = self.handle[0]
        
        # Create empty tensor on the correct device
        t = torch.tensor([], device=device, dtype=self.dtype)
        
        # Set the tensor to use the shared storage
        t.set_(storage)
        
        # Reshape to original shape
        return t.view(self.shape)
    
    def __repr__(self) -> str:
        return (f"CudaIPCWrapper(shape={self.shape}, dtype={self.dtype}, "
                f"device={self.device})")


__all__ = ["CudaIPCWrapper"]

