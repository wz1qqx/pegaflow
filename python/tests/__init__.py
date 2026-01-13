"""PegaFlow connector integration test suite.

Requirements:
- Build Rust extension: maturin develop --release
- GPU available for PegaServer

Run tests:
    cd python && pytest tests/ -v

The pega_server fixture automatically starts/stops PegaServer.
"""
