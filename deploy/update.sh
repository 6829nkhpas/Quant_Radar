#!/bin/sh
# Pull the newest radar image and restart it. The ONLY thing CI's deploy key can run:
# installed on dev-vm as /opt/quant-radar/update.sh (root:root 755), allowed for the
# `radar-deploy` user by /etc/sudoers.d/radar-deploy, and forced by that user's
# authorized_keys (`restrict,command="sudo -n /opt/quant-radar/update.sh"`), so the
# command CI sends is ignored. See README "One-time VM setup".
set -eu
cd /opt/quant-radar
docker compose pull -q
docker compose up -d
# Only this repo's superseded images; the research API's images are left alone.
docker image ls ghcr.io/6829nkhpas/quant_radar --filter dangling=true -q | xargs -r docker rmi
docker compose ps --format '{{.Name}} {{.Image}} {{.Status}}'
