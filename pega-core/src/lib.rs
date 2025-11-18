use std::{collections::HashMap, sync::Arc};

use cudarc::driver::CudaContext;

pub struct PegaEngine {
    context: Arc<CudaContext>,
    /// Store registered KV cache pointers (new IPC wrapper): layer_name -> KVCachePtr
    kv_cache_ptrs: HashMap<String, KVCachePtr>,
    /// Store saved KV blocks: (layer_name, block_hash) -> block_data
    kv_storage: HashMap<(String, Vec<u8>), Vec<u8>>,
}

/// Represents a CUDA IPC handle for a KV cache tensor (legacy)
#[derive(Debug, Clone)]
pub struct KVCacheHandle {
    pub device: i32,
    pub ipc_handle: Vec<u8>,
    pub size: usize,
    pub offset: usize,
    pub ipc_event_handle: Vec<u8>,
}

/// Represents a GPU memory pointer for a KV cache tensor (new IPC wrapper method)
#[derive(Debug, Clone)]
pub struct KVCachePtr {
    pub data_ptr: u64,
    pub size_bytes: usize,
}

impl PegaEngine {
    /// Create a new PegaEngine instance
    pub fn new() -> Self {
        // default device is 0
        let context = cudarc::driver::CudaContext::new(0).unwrap();
        PegaEngine {
            context,
            kv_cache_ptrs: HashMap::new(),
            kv_storage: HashMap::new(),
        }
    }

    /// Register a KV cache by GPU pointer (new IPC wrapper method)
    ///
    /// This method stores the GPU pointer that was reconstructed from an IPC handle
    /// in Python. The pointer can be used directly for GPU memory operations.
    ///
    /// Args:
    ///   - layer_name: Name of the layer
    ///   - data_ptr: GPU data pointer
    ///   - size_bytes: Size of the tensor in bytes
    pub fn register_kv_cache_ptr(&mut self, layer_name: String, data_ptr: u64, size_bytes: usize) {
        let cache_ptr = KVCachePtr {
            data_ptr,
            size_bytes,
        };

        self.kv_cache_ptrs.insert(layer_name, cache_ptr);
    }

    /// Unregister all KV cache handles
    pub fn unregister_all_kv_caches(&mut self) {
        self.kv_cache_ptrs.clear();
    }

    /// Get the number of registered KV caches
    pub fn num_registered_kv_caches(&self) -> usize {
        println!("self.kv_cache_handles.len() = {}", self.kv_cache_ptrs.len());
        self.kv_cache_ptrs.len()
    }

    pub fn save_kv_blocks_from_ipc(
        &mut self,
        _layer_name: String,
        _block_ids: Vec<i32>,
        _block_hashes: Vec<Vec<u8>>,
    ) -> Result<(), String> {
        Ok(())
    }

    /// Copy data from GPU to CPU
    fn copy_gpu_to_cpu(
        &self,
        _stream: &cudarc::driver::CudaStream,
        gpu_base_ptr: u64,
        offset: usize,
        cpu_buffer: &mut [u8],
        size: usize,
    ) -> Result<(), String> {
        use cudarc::driver::sys;

        let src_ptr = gpu_base_ptr + offset as u64;
        let dst_ptr = cpu_buffer.as_mut_ptr();

        unsafe {
            // Use synchronous copy for simplicity
            let result = sys::cuMemcpyDtoH_v2(dst_ptr as *mut std::ffi::c_void, src_ptr, size);
            if result != sys::cudaError_enum::CUDA_SUCCESS {
                return Err(format!("cuMemcpyDtoH failed: {:?}", result));
            }
        }

        Ok(())
    }

    /// Get storage statistics
    /// Returns (num_blocks, total_bytes)
    pub fn get_storage_stats(&self) -> (usize, usize) {
        let num_blocks = self.kv_storage.len();
        let total_bytes: usize = self.kv_storage.values().map(|v| v.len()).sum();
        (num_blocks, total_bytes)
    }

    /// Check which KV blocks are available in CPU storage
    ///
    /// Args:
    ///   - layer_name: Name of the layer
    ///   - block_hashes: List of block hashes to check
    ///
    /// Returns:
    ///   - Vec<bool>: For each hash, true if available in storage
    pub fn check_kv_blocks_availability(
        &self,
        layer_name: String,
        block_hashes: Vec<Vec<u8>>,
    ) -> Vec<bool> {
        println!("\n=== Rust: check_kv_blocks_availability ===");
        println!("Layer: {}", layer_name);
        println!("Checking {} block hashes", block_hashes.len());

        let mut availability = Vec::with_capacity(block_hashes.len());

        for (idx, block_hash) in block_hashes.iter().enumerate() {
            let key = (layer_name.clone(), block_hash.clone());
            let available = self.kv_storage.contains_key(&key);
            availability.push(available);

            if available {
                println!(
                    "  Block {} (hash {:?}): AVAILABLE",
                    idx,
                    &block_hash[..8.min(block_hash.len())]
                );
            } else {
                println!(
                    "  Block {} (hash {:?}): NOT FOUND",
                    idx,
                    &block_hash[..8.min(block_hash.len())]
                );
            }
        }

        let num_available = availability.iter().filter(|&&x| x).count();
        println!(
            "=== Result: {}/{} blocks available ===\n",
            num_available,
            block_hashes.len()
        );

        availability
    }

    /// Load KV blocks from CPU memory to GPU via IPC handle
    ///
    /// Args:
    ///   - layer_name: Name of the layer
    ///   - block_ids: GPU block IDs to load into
    ///   - block_hashes: Content hashes for each block
    pub fn load_kv_blocks_to_ipc(
        &self,
        layer_name: String,
        block_ids: Vec<i32>,
        block_hashes: Vec<Vec<u8>>,
    ) -> Result<(), String> {
        Ok(())
    }

    /// Copy data from CPU to GPU
    fn copy_cpu_to_gpu(
        &self,
        _stream: &cudarc::driver::CudaStream,
        gpu_base_ptr: u64,
        offset: usize,
        cpu_buffer: &[u8],
        size: usize,
    ) -> Result<(), String> {
        use cudarc::driver::sys;

        if cpu_buffer.len() < size {
            return Err(format!(
                "CPU buffer too small: {} bytes, need {} bytes",
                cpu_buffer.len(),
                size
            ));
        }

        let dst_ptr = gpu_base_ptr + offset as u64;
        let src_ptr = cpu_buffer.as_ptr();

        unsafe {
            // Use synchronous copy for simplicity
            let result = sys::cuMemcpyHtoD_v2(dst_ptr, src_ptr as *const std::ffi::c_void, size);
            if result != sys::cudaError_enum::CUDA_SUCCESS {
                return Err(format!("cuMemcpyHtoD failed: {:?}", result));
            }
        }

        Ok(())
    }
}

impl Default for PegaEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cudarc_basic() {
        // Get a stream for GPU 0
        let ctx = cudarc::driver::CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();

        // copy a rust slice to the device
        let _inp = stream.clone_htod(&[1.0f32; 100]).unwrap();

        // or allocate directly
        let _out = stream.alloc_zeros::<f32>(100).unwrap();
    }

    #[test]
    fn test_gpu_to_cpu_copy() {
        // 1. Create context and stream
        let ctx = cudarc::driver::CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();

        // 2. Allocate and initialize data on GPU
        let test_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let gpu_data = stream.clone_htod(&test_data).unwrap();

        // 3. Copy from GPU to CPU
        let cpu_data: Vec<f32> = stream.clone_dtoh(&gpu_data).unwrap();

        // 4. Verify the data
        assert_eq!(cpu_data, test_data);
        println!("GPU->CPU copy test passed! Data: {:?}", cpu_data);
    }

    #[test]
    fn test_gpu_to_cpu_copy_bf16() {
        // 1. Create context and stream
        let ctx = cudarc::driver::CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();

        // 2. Simulate KV cache block: [2, 16, 12, 64] * bf16 (2 bytes)
        let block_size = 2 * 16 * 12 * 64;
        let test_data: Vec<u8> = (0..block_size).map(|i| (i % 256) as u8).collect();

        // 3. Copy to GPU
        let gpu_block = stream.clone_htod(&test_data).unwrap();

        // 4. Copy back to CPU
        let cpu_block: Vec<u8> = stream.clone_dtoh(&gpu_block).unwrap();

        // 5. Verify
        assert_eq!(cpu_block.len(), block_size);
        assert_eq!(cpu_block, test_data);
        println!(
            "GPU->CPU BF16 block copy test passed! Block size: {} bytes",
            block_size
        );
    }
}
