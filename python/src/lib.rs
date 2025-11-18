use pega_core::PegaEngine as CoreEngine;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

/// Python wrapper for PegaEngine
#[pyclass]
struct PegaEngine {
    engine: CoreEngine,
}

#[pymethods]
impl PegaEngine {
    /// Create a new PegaEngine instance
    #[new]
    fn new() -> Self {
        PegaEngine {
            engine: CoreEngine::new(),
        }
    }

    /// Register a KV cache by GPU pointer (new IPC wrapper method)
    ///
    /// This method receives a GPU pointer that was reconstructed from an IPC handle
    /// in Python. The Python side handles the IPC serialization/deserialization and
    /// tensor reconstruction, then passes the GPU pointer to Rust.
    ///
    /// Args:
    ///     layer_name: Name of the layer
    ///     data_ptr: GPU data pointer (as u64)
    ///     size_bytes: Size of the tensor in bytes
    fn register_kv_cache_ptr(&mut self, layer_name: String, data_ptr: u64, size_bytes: usize) {
        self.engine
            .register_kv_cache_ptr(layer_name, data_ptr, size_bytes);
    }

    /// Unregister all KV cache handles
    fn unregister_all_kv_caches(&mut self) {
        self.engine.unregister_all_kv_caches();
    }

    /// Get the number of registered KV caches
    fn num_registered_kv_caches(&self) -> usize {
        self.engine.num_registered_kv_caches()
    }

    /// Save KV blocks from GPU via IPC handle to CPU memory
    ///
    /// Args:
    ///     layer_name: Name of the layer
    ///     block_ids: GPU block IDs to copy (list of ints)
    ///     block_hashes: Content hashes for each block (list of bytes)
    fn save_kv_blocks_from_ipc(
        &mut self,
        layer_name: String,
        block_ids: Vec<i32>,
        block_hashes: Vec<Vec<u8>>,
    ) -> PyResult<()> {
        self.engine
            .save_kv_blocks_from_ipc(layer_name, block_ids, block_hashes)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
    }

    /// Get storage statistics
    /// Returns (num_blocks, total_bytes)
    fn get_storage_stats(&self) -> (usize, usize) {
        self.engine.get_storage_stats()
    }

    /// Check which KV blocks are available in CPU storage
    ///
    /// Args:
    ///     layer_name: Name of the layer
    ///     block_hashes: List of block hashes to check (list of bytes)
    ///
    /// Returns:
    ///     List of booleans indicating availability for each block
    fn check_kv_blocks_availability(
        &self,
        layer_name: String,
        block_hashes: Vec<Vec<u8>>,
    ) -> Vec<bool> {
        self.engine
            .check_kv_blocks_availability(layer_name, block_hashes)
    }

    /// Load KV blocks from CPU memory to GPU via IPC handle
    ///
    /// Args:
    ///     layer_name: Name of the layer
    ///     block_ids: GPU block IDs to load into (list of ints)
    ///     block_hashes: Content hashes for each block (list of bytes)
    fn load_kv_blocks_to_ipc(
        &self,
        layer_name: String,
        block_ids: Vec<i32>,
        block_hashes: Vec<Vec<u8>>,
    ) -> PyResult<()> {
        self.engine
            .load_kv_blocks_to_ipc(layer_name, block_ids, block_hashes)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
    }
}

/// A Python module implemented in Rust.
/// This module is named "pegaflow" and will be imported as: from pegaflow import PegaEngine
#[pymodule]
fn pegaflow(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PegaEngine>()?;
    Ok(())
}
