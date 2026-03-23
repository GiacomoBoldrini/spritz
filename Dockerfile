#!/usr/bin/env -S docker build . --tag=gpizzati/spritz:latest --file

FROM gitlab-registry.cern.ch/linuxsupport/alma9-base:latest
ADD docker-files/batch9-stable.repo /etc/yum.repos.d/

# Update system and install dev tools
RUN dnf -y update && dnf clean all

RUN dnf install -y dnf-plugins-core \
 && dnf config-manager --set-enabled crb \
 && dnf clean all


#RUN dnf config-manager --set-enabled crb && \
#    dnf -y --allowerasing --skip-broken install \
#    myschedd krb5-workstation ngbauth-submit perl-Sys-Syslog gcc gcc-c++ cmake git make curl wget bzip2 which tar --exclude=openssh && \
#    dnf clean all

# STEP 5: Pulizia Totale - Installiamo solo il minimo indispensabile
RUN echo "INSTALLAZIONE MINIMALE" && \
    dnf install -y epel-release && \
    /usr/bin/crb enable && \
    # Installiamo i compilatori ma ESCLUDIAMO git e condor che tirano dentro SSH
    dnf -y --allowerasing install \
     krb5-workstation \
     perl-Sys-Syslog \
     gcc gcc-c++ cmake make curl wget bzip2 which tar \
     --exclude=openssh* --exclude=git* && \
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

RUN mamba create -n spritz --file env.yaml --yes --no-pin && \
    source /usr/local/etc/profile.d/conda.sh && \
    conda activate spritz && \
    # Installiamo setuptools vecchio PRIMA di tutto il resto
    pip install "setuptools<70" && \
    # Ora installiamo rucio e gli altri SENZA isolamento, così usano il setuptools sopra
    PIP_NO_BUILD_ISOLATION=0 pip install --no-cache-dir rucio-clients==35.5.0 && \
    pip install --no-cache-dir correctionlib git+https://github.com/giorgiopizz/correctionlib && \
    echo "Done installing packages"


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
# VOMS configuration
# ==========================

RUN echo "START VOMS" && mkdir -p /etc/grid-security /usr/local/etc/grid-security && \
    curl -sL https://github.com/opensciencegrid/osg-vo-config/archive/refs/heads/master.tar.gz -o /tmp/osg.tar.gz && \
    mkdir -p /tmp/osg-unpacked && \
    tar -xzf /tmp/osg.tar.gz -C /tmp/osg-unpacked --strip-components=1 && \
    cp -r /tmp/osg-unpacked/vomsdir /etc/grid-security/ && \
    if [ -e /tmp/osg-unpacked/vomses ]; then cp -r /tmp/osg-unpacked/vomses /etc/; fi && \
    ln -sf /etc/grid-security /usr/local/etc/grid-security && \
    rm -rf /tmp/osg.tar.gz /tmp/osg-unpacked

# ==========================
# Condor configuration
# ==========================

WORKDIR /

RUN mkdir -p /usr/local/etc/condor/config.d
ADD docker-files/condor_submit.config /usr/local/etc/condor/config.d/

# but config still looked for in /etc/condor for some reason
RUN mkdir -p /etc/condor/
ADD docker-files/condor_config /etc/condor/
RUN mkdir -p /etc/condor/config.d
ADD docker-files/condor_submit.config /etc/condor/config.d/

# ==========================
# Final configuration
# ==========================
WORKDIR /workspace

# Default shell with GCC and conda activated
SHELL ["/bin/bash", "-c"]
ENTRYPOINT ["/bin/bash"]
