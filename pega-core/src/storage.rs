use hashlink::LruCache;
use std::collections::HashMap;
use std::ops::{Deref, DerefMut};
use std::sync::{Arc, Mutex, MutexGuard};

use crate::pinned_pool::{PinnedAllocation, PinnedMemoryPool};

const RECLAIM_BATCH_OBJECTS: usize = 256;

// A "layer" in this file always refers to a model layer registered by vLLM
// (e.g. transformer block 0, 1, ...). vLLM reports the total number of layers
// up front and that count never changes for the lifetime of the engine, so
// we treat it as immutable. Any attempt to resize the per-layer vector means
// the registration logic violated this contract and should panic.

pub type BlockHash = Vec<u8>;
type LayerBlocks = Vec<Option<Arc<Block>>>;

struct LayerBlocksState {
    blocks: LayerBlocks,
    is_complete: bool,
}

impl LayerBlocksState {
    fn new(num_layers: usize) -> Self {
        Self {
            blocks: vec![None; num_layers],
            is_complete: false,
        }
    }

    fn mark_layer_ready(&mut self, layer_id: usize, block: Arc<Block>) {
        assert!(
            layer_id < self.blocks.len(),
            "layer_id {} out of bounds ({} layers)",
            layer_id,
            self.blocks.len()
        );
        self.blocks[layer_id] = Some(block);
        if layer_id == self.blocks.len() - 1 {
            self.is_complete = true;
        }
    }
}

/// Wrapper for per-layer block vectors with a fixed weight for cache eviction.
pub struct LayerBlocksWithWeight {
    inner: Mutex<LayerBlocksState>,
}

impl LayerBlocksWithWeight {
    pub fn new(num_layers: usize) -> Self {
        Self {
            inner: Mutex::new(LayerBlocksState::new(num_layers)),
        }
    }

    pub fn lock_blocks(&self) -> LayerBlocksGuard<'_> {
        LayerBlocksGuard {
            inner: self.lock_state(),
        }
    }

    pub fn is_complete(&self) -> bool {
        self.inner
            .lock()
            .expect("layer blocks lock poisoned")
            .is_complete
    }

    fn lock_state(&self) -> MutexGuard<'_, LayerBlocksState> {
        self.inner.lock().expect("layer blocks lock poisoned")
    }
}

pub struct LayerBlocksGuard<'a> {
    inner: MutexGuard<'a, LayerBlocksState>,
}

impl<'a> Deref for LayerBlocksGuard<'a> {
    type Target = LayerBlocks;

    fn deref(&self) -> &Self::Target {
        &self.inner.blocks
    }
}

impl<'a> DerefMut for LayerBlocksGuard<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner.blocks
    }
}

impl<'a> LayerBlocksGuard<'a> {
    pub fn mark_layer_ready(&mut self, layer_id: usize, block: Arc<Block>) {
        self.inner.mark_layer_ready(layer_id, block);
    }
}

/// CPU block data stored in pinned memory.
pub struct Block {
    /// Pointer to K segment (or combined data if contiguous)
    k_ptr: std::ptr::NonNull<u8>,
    /// Pointer to V segment (if stored separately)
    v_ptr: Option<std::ptr::NonNull<u8>>,
    size: usize,
    /// Shared RAII allocation handle for K memory (automatically freed when last reference drops)
    #[allow(dead_code)]
    k_allocation: Arc<PinnedAllocation>,
    /// Shared RAII allocation handle for V memory (if separate from K)
    #[allow(dead_code)]
    v_allocation: Option<Arc<PinnedAllocation>>,
}

impl Block {
    pub fn new_contiguous(ptr: *mut u8, size: usize, allocation: Arc<PinnedAllocation>) -> Self {
        Self {
            k_ptr: unsafe { std::ptr::NonNull::new_unchecked(ptr) },
            v_ptr: None,
            size,
            k_allocation: allocation,
            v_allocation: None,
        }
    }

    pub fn new_split(
        k_ptr: *mut u8,
        v_ptr: *mut u8,
        size: usize,
        k_allocation: Arc<PinnedAllocation>,
        v_allocation: Arc<PinnedAllocation>,
    ) -> Self {
        Self {
            k_ptr: unsafe { std::ptr::NonNull::new_unchecked(k_ptr) },
            v_ptr: Some(unsafe { std::ptr::NonNull::new_unchecked(v_ptr) }),
            size,
            k_allocation,
            v_allocation: Some(v_allocation),
        }
    }

    pub fn k_ptr(&self) -> *mut u8 {
        self.k_ptr.as_ptr()
    }

    pub fn v_ptr(&self) -> Option<*mut u8> {
        self.v_ptr.map(|ptr| ptr.as_ptr())
    }

    pub fn size(&self) -> usize {
        self.size
    }
}

// Safety: pinned memory ownership is tracked by Arc counters on the allocations.
unsafe impl Send for Block {}
unsafe impl Sync for Block {}

pub struct StorageEngine {
    kv_storage: Mutex<LruCache<BlockHash, Arc<ContextualLayerBlocks>>>,
    pinned_pool: Arc<PinnedMemoryPool>,
    context_layer_counts: Mutex<HashMap<String, usize>>,
}

impl StorageEngine {
    pub fn new(capacity_bytes: usize) -> Self {
        let pinned_pool = Arc::new(PinnedMemoryPool::new(capacity_bytes));
        // Use the pool capacity as the cache weight to keep eviction proportional to bytes.
        let cache_capacity = capacity_bytes.max(1);
        let kv_storage = Mutex::new(LruCache::new(cache_capacity));

        Self {
            kv_storage,
            pinned_pool,
            context_layer_counts: Mutex::new(HashMap::new()),
        }
    }

    pub fn allocate(&self, size: usize) -> Arc<PinnedAllocation> {
        loop {
            if let Some(allocation) = self.pinned_pool.allocate(size) {
                return Arc::new(allocation);
            }

            let reclaimed = self.reclaim(RECLAIM_BATCH_OBJECTS);
            if reclaimed > 0 {
                continue;
            } else {
                let (used, total) = self.pinned_pool.usage();
                panic!(
                    "Pinned memory pool exhausted! Requested: {:.2} MB, Used: {:.2} MB, Total: {:.2} MB, Cache empty",
                    size as f64 / 1e6,
                    used as f64 / 1e6,
                    total as f64 / 1e6
                );
            }
        }
    }

    fn reclaim(&self, target_objects: usize) -> usize {
        if target_objects == 0 {
            return 0;
        }

        let mut freed_entries = 0;
        let mut cache = self.kv_storage.lock().unwrap();

        while freed_entries < target_objects {
            let Some((_hash, _layer_blocks)) = cache.remove_lru() else {
                break;
            };
            freed_entries += 1;
        }

        freed_entries
    }

    pub fn initialize_layer_count(&self, context_id: &str, total_layers: usize) {
        if total_layers == 0 {
            panic!("total_layers must be > 0");
        }
        let mut guard = self.context_layer_counts.lock().unwrap();
        match guard.get(context_id) {
            Some(existing) if *existing != total_layers => {
                panic!(
                    "layer count changed for context {}: expected {}, got {}",
                    context_id, existing, total_layers
                )
            }
            Some(_) => {}
            None => {
                guard.insert(context_id.to_string(), total_layers);
            }
        }
    }

    fn get_layer_count(&self, context_id: &str) -> usize {
        *self
            .context_layer_counts
            .lock()
            .unwrap()
            .get(context_id)
            .unwrap_or_else(|| panic!("layer count not initialized for context {}", context_id))
    }

    pub fn layer_has_block(&self, context_id: &str, block_hash: &[u8], layer_id: usize) -> bool {
        let mut cache = self.kv_storage.lock().unwrap();
        cache
            .get(block_hash)
            .and_then(|entry| entry.get_context(context_id))
            .map(|blocks| {
                let blocks = blocks.lock_blocks();
                blocks.get(layer_id).and_then(|opt| opt.as_ref()).is_some()
            })
            .unwrap_or(false)
    }

    pub fn block_is_complete(&self, context_id: &str, block_hash: &[u8]) -> bool {
        let mut cache = self.kv_storage.lock().unwrap();
        cache
            .get(block_hash)
            .and_then(|entry| entry.get_context(context_id))
            .map(|blocks| blocks.is_complete())
            .unwrap_or(false)
    }

    pub fn insert_block(
        &self,
        context_id: &str,
        block_hash: BlockHash,
        layer_id: usize,
        block: Arc<Block>,
    ) {
        let total_layers = self.get_layer_count(context_id);
        let mut cache = self.kv_storage.lock().unwrap();
        let entry = cache.get(&block_hash).cloned().unwrap_or_else(|| {
            let new_blocks = Arc::new(ContextualLayerBlocks::new());
            cache.insert(block_hash.clone(), Arc::clone(&new_blocks));
            new_blocks
        });
        let context_blocks = entry.get_or_insert_context(context_id, total_layers);
        let mut blocks = context_blocks.lock_blocks();
        blocks.mark_layer_ready(layer_id, block);
    }

    pub fn lookup_many_for_context(
        &self,
        context_id: &str,
        block_hashes: &[Vec<u8>],
    ) -> Result<Vec<Arc<LayerBlocksWithWeight>>, String> {
        let mut cache = self.kv_storage.lock().unwrap();
        let mut result = Vec::with_capacity(block_hashes.len());
        for hash in block_hashes {
            let layer_blocks = cache
                .get(hash)
                .cloned()
                .ok_or_else(|| "Missing KV block hash".to_string())?;
            let context_blocks = layer_blocks
                .get_context(context_id)
                .ok_or_else(|| "Missing KV blocks for context".to_string())?;
            result.push(context_blocks);
        }
        Ok(result)
    }

    pub fn clear_context(&self, context_id: &str) {
        self.context_layer_counts.lock().unwrap().remove(context_id);

        let mut cache = self.kv_storage.lock().unwrap();
        let keys: Vec<BlockHash> = cache.iter().map(|(k, _)| k.clone()).collect();
        for key in keys {
            let should_remove = if let Some(entry) = cache.get(&key) {
                if entry.remove_context(context_id) {
                    entry.is_empty()
                } else {
                    false
                }
            } else {
                false
            };
            if should_remove {
                cache.remove(&key);
            }
        }
    }
}

struct ContextualLayerBlocks {
    contexts: Mutex<HashMap<String, Arc<LayerBlocksWithWeight>>>,
}

impl ContextualLayerBlocks {
    fn new() -> Self {
        Self {
            contexts: Mutex::new(HashMap::new()),
        }
    }

    fn get_or_insert_context(
        &self,
        context_id: &str,
        total_layers: usize,
    ) -> Arc<LayerBlocksWithWeight> {
        let mut guard = self.contexts.lock().expect("context map poisoned");
        guard
            .entry(context_id.to_string())
            .or_insert_with(|| Arc::new(LayerBlocksWithWeight::new(total_layers)))
            .clone()
    }

    fn get_context(&self, context_id: &str) -> Option<Arc<LayerBlocksWithWeight>> {
        self.contexts
            .lock()
            .expect("context map poisoned")
            .get(context_id)
            .cloned()
    }

    fn remove_context(&self, context_id: &str) -> bool {
        self.contexts
            .lock()
            .expect("context map poisoned")
            .remove(context_id)
            .is_some()
    }

    fn is_empty(&self) -> bool {
        self.contexts
            .lock()
            .expect("context map poisoned")
            .is_empty()
    }
}
