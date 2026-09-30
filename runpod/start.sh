#!/usr/bin/env bash
# runpod/start.sh -- pod entrypoint for the RunPod template (container start command).
#
# Our image is built from nvidia/cuda, not a Runpod base, so nothing starts sshd for
# us. Without this you are locked out: no SSH and no web terminal. It reproduces what
# runpod/pytorch's /start.sh does (PUBLIC_KEY -> authorized_keys, host keys, sshd),
# then idles so the pod stays up for tmux/ssh sessions. Requires openssh-server in
# the image (k8s/Dockerfile.pipeline) and 22/tcp exposed on the template.
mkdir -p ~/.ssh && chmod 700 ~/.ssh
if [ -n "${PUBLIC_KEY:-}" ]; then
  echo "$PUBLIC_KEY" >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys
fi
mkdir -p /run/sshd
ssh-keygen -A
/usr/sbin/sshd
echo "sshd up; pod $RUNPOD_POD_ID idle -- attach with ssh or the web terminal"
exec sleep infinity
