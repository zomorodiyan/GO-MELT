FROM nvcr.io/nvidia/pytorch:24.10-py3

WORKDIR /workspace

# Install system utilities for monitoring and parallel execution
RUN apt-get update && apt-get install -y \
    tmux \
    screen \
    vim \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Clone dotfiles configuration (tmux, vim, etc.)
RUN git clone https://github.com/zomorodiyan/ubuntu.git /root/.dotfiles && \
    ln -sf /root/.dotfiles/.tmux.conf /root/.tmux.conf && \
    ln -sf /root/.dotfiles/.vimrc /root/.vimrc && \
    mkdir -p /root/.vim/undodir

# Install GO-MELT dependencies
RUN pip install --no-cache-dir \
    "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
    pyevtk \
    dill

CMD ["/bin/bash"]
