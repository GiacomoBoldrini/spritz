#!/usr/bin/env -S docker build . --tag=gpizzati/spritz:latest --file

FROM gitlab-registry.cern.ch/linuxsupport/alma9-base:latest

# Update system and install dev tools
RUN dnf -y update && dnf clean all

RUN dnf install -y dnf-plugins-core \
 && dnf config-manager --set-enabled crb \
 && dnf clean all

RUN dnf config-manager --set-enabled crb && \
    dnf -y --allowerasing --skip-broken install \
        gcc gcc-c++ cmake git make curl wget bzip2 which tar --exclude=openssh && \
    dnf clean all

# Install Mambaforge (better than Miniforge for speed)
WORKDIR /tmp
RUN curl -sSL "https://github.com/conda-forge/miniforge/releases/download/23.3.1-1/Mambaforge-Linux-x86_64.sh" -o mambaforge.sh && \
    bash mambaforge.sh -bfp /usr/local && \
    rm -f mambaforge.sh && \
    mamba update -n base mamba -y && \
    mamba clean --all --yes

# Activate toolset, build conda env
COPY env.yaml env.yaml

RUN mamba create -n spritz --file env.yaml --yes && \
    source /usr/local/etc/profile.d/conda.sh && \
    conda activate spritz && \
    pip install --no-cache-dir correctionlib git+https://github.com/giorgiopizz/correctionlib && \
    echo "Done installing packages"
    # mamba clean --all -f -y

# VOMS / grid setup
RUN mkdir -p /usr/local/etc/grid-security && \
    ln -s /usr/local/etc/grid-security /etc/grid-security && \
    curl -L https://github.com/opensciencegrid/osg-vo-config/archive/refs/heads/master.tar.gz | \
    tar -xz --strip-components=1 --directory=/usr/local/etc/grid-security --wildcards "*/vomses" "*/vomsdir" && \
    mv /usr/local/etc/grid-security/vomses /etc

# ==========================
# Install spritz package
# ==========================
# Copy your source code and install

WORKDIR /opt/spritz

# Copy minimal files for installation
COPY README.md pyproject.toml requirements.txt start.sh ./
COPY src ./src

COPY data ./data

# Verify the conda env exists
RUN mamba info --envs

ENV PATH=/usr/local/envs/spritz/bin:$PATH

RUN SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SPRITZ=0.1.0 \
    SETUPTOOLS_SCM_PRETEND_VERSION=0.1.0 \
    /usr/local/envs/spritz/bin/pip install --no-cache-dir .

# Make spritz CLI tools globally available
RUN ln -s /usr/local/envs/spritz/bin/spritz-* /usr/local/bin/ || true

# Optional: Set SPRITZ_PATH and PYTHONPATH
ENV SPRITZ_PATH=/opt/spritz
ENV PYTHONPATH=/opt/spritz:$PYTHONPATH

# ==========================
# Integrate start.sh
# ==========================
# Copy start.sh into the image
COPY start.sh /opt/spritz/start.sh
RUN chmod +x /opt/spritz/start.sh

# Source start.sh for every interactive shell
RUN echo "source /opt/spritz/start.sh" >> /etc/bash.bashrc

# ==========================
# Final configuration
# ==========================
WORKDIR /workspace

# Default shell with GCC and conda activated
SHELL ["/bin/bash", "-c"]
ENTRYPOINT ["/bin/bash"]
