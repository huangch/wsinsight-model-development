#!/bin/sh
# Built from the parent directory: the Dockerfile installs the sibling
# kurtorank checkout from source, so the build context must contain both.
# The container uid/gid is chosen at RUN time by the image entrypoint (it
# remaps the in-image "user" to the owner of the mounted /workspace, or to
# $HOST_UID/$HOST_GID), so the build never bakes the caller's id.
docker build -f ./Dockerfile -t wsitrain:latest ..
docker tag wsitrain:latest huangchtw/wsitrain:latest
docker push huangchtw/wsitrain:latest
