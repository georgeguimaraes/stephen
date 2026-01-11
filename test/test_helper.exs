# Configure EXLA as the default backend for faster computation
Nx.global_default_backend(EXLA.Backend)

ExUnit.start(exclude: [:slow, :integration])
