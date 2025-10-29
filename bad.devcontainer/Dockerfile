FROM ghcr.io/prefix-dev/pixi:noble-cuda-12.8.1

# Important packages
RUN apt-get update && apt-get install -y \
    vim nano git sudo make \
    locales xdg-utils wget \
    nvtop htop tmux curl

# Set the locale. Required for TMUX and neovim
RUN sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && locale-gen
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

# Change user 
ARG UID
ARG GID
ARG UNAME
RUN useradd -m -l -s /bin/bash -d /home/"${UNAME}" -u "${UID}" -g "${GID}" "${UNAME}" && \
    echo "${UNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
USER "${UNAME}"
CMD [ "tail", "-f", "/dev/null" ]
