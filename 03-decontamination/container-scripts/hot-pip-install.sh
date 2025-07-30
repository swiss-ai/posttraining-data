if [ -n "${SLURM_ONE_ENTRYPOINT_SCRIPT_PER_JOB}" ] && [ "${SLURM_PROCID}" -gt 0 ]; then
  sleep 60
elif [ -n "${SLURM_ONE_ENTRYPOINT_SCRIPT_PER_NODE}" ] && [ "${SLURM_LOCALID}" -gt 0 ]; then
  sleep 60
else
  echo "[HOT PIP INSTALL] installing additional packages on SLURM_PROCID ${SLURM_PROCID}, SLURM_LOCALID ${SLURM_LOCALID}, hostname $(hostname)."
  pushd data/shared/pip-wheels/
  popd
fi

# Assert everything is installed correctly
python - <<EOF
import sys, ${PACKAGE_NAME}, transformers
if transformers.__version__ == "4.50.0.dev0":
    sys.exit(0)
else:
    print("Did not find the right version")
    sys.exit(1)
EOF

echo "SLURM_NODEID $SLURM_NODEID SLURM_PROCID $SLURM_PROCID SLURM_LOCALID $SLURM_LOCALID done with hot-pip-install.sh"
