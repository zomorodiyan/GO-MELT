FROM nvcr.io/nvidia/pytorch:24.10-py3

WORKDIR /workspace

# Install GO-MELT dependencies
RUN pip install --no-cache-dir \
    "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
    pyevtk \
    dill

CMD ["/bin/bash"]
