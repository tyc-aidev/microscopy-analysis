#!/usr/bin/env bash
# Thin wrapper around `gcloud compute` for the Sprint 0 T4 reproduction host.
# Full runbook + rationale: configs/cloud/gcp_t4_repro.md (issue #16).
#
# Usage:
#   scripts/cloud_gcp_t4.sh create     # spot n1-standard-4 + 1x T4 (Deep Learning VM, cu113)
#   scripts/cloud_gcp_t4.sh ssh        # ssh into the instance
#   scripts/cloud_gcp_t4.sh delete     # delete the instance (stop billing)
#   scripts/cloud_gcp_t4.sh status     # list compute instances
#
# Env overrides: INSTANCE, ZONE, MACHINE_TYPE, PROJECT.
set -euo pipefail

# The bundled gcloud breaks under Python 3.12 ("No module named 'imp'"); pin an
# older interpreter unless the caller already set CLOUDSDK_PYTHON.
: "${CLOUDSDK_PYTHON:=$(command -v python3.11 || command -v python3.10 || command -v python3.9 || true)}"
export CLOUDSDK_PYTHON

INSTANCE="${INSTANCE:-micronet-repro-t4}"
ZONE="${ZONE:-us-central1-b}"
MACHINE_TYPE="${MACHINE_TYPE:-n1-standard-4}"

gc() { gcloud "$@" ${PROJECT:+--project="$PROJECT"}; }

case "${1:-}" in
  create)
    gc compute instances create "$INSTANCE" \
      --zone="$ZONE" \
      --machine-type="$MACHINE_TYPE" \
      --accelerator=type=nvidia-tesla-t4,count=1 \
      --image-family=common-cu113-debian-11 \
      --image-project=deeplearning-platform-release \
      --maintenance-policy=TERMINATE \
      --provisioning-model=SPOT \
      --instance-termination-action=DELETE \
      --boot-disk-size=100GB \
      --boot-disk-type=pd-balanced \
      --metadata="install-nvidia-driver=True"
    echo "Created $INSTANCE in $ZONE. Next: $0 ssh"
    ;;
  ssh)
    gc compute ssh "$INSTANCE" --zone="$ZONE"
    ;;
  delete)
    gc compute instances delete "$INSTANCE" --zone="$ZONE" --quiet
    echo "Deleted $INSTANCE."
    ;;
  status)
    gc compute instances list
    ;;
  *)
    echo "usage: $0 {create|ssh|delete|status}" >&2
    exit 2
    ;;
esac
