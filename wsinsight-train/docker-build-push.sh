#!/bin/sh
# Built from the parent directory: the Dockerfile installs the sibling
# kurtorank checkout from source, so the build context must contain both.
# The container uid/gid is chosen at RUN time by the image entrypoint (it
# remaps the in-image "user" to the owner of the mounted /workspace, or to
# $HOST_UID/$HOST_GID), so the build never bakes the caller's id.
# Run from this directory; the context is the parent so both wsinsight-train
# and its sibling kurtorank are visible to COPY. ../.dockerignore keeps the
# 3.9 TB of data/ and models/ out of it.
docker build -f ./Dockerfile -t wsitrain:latest ..
docker tag wsitrain:latest huangchtw/wsitrain:latest
docker push huangchtw/wsitrain:latest
