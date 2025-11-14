FROM mcr.microsoft.com/devcontainers/python:3.10

ARG _DEVCONTAINER_FEATURES
RUN ${_DEVCONTAINER_FEATURES}

RUN apt-get update && apt-get install -y libgl1
