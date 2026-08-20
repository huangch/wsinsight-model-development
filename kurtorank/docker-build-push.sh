#!/bin/sh
# The container uid/gid is chosen at RUN time by the image entrypoint (it
# remaps the in-image "user" to the owner of the mounted /workspace, or to
# $HOST_UID/$HOST_GID), so the build never bakes the caller's id.
docker build -f ./Dockerfile -t kurtorank:latest .
docker tag kurtorank:latest huangchtw/kurtorank:latest
docker push huangchtw/kurtorank:latest
