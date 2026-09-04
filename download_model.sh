#!/usr/bin/env bash
# Downloads the base model for OneAMR (ADTC 2026 Gate 1).
# Base model: Qwen2.5-3B-Instruct, quantized to GGUF Q4_K_M by bartowski.
# ~1.9 GB on disk, leaves comfortable headroom under the 7 GB inference cap.

set -euo pipefail

MODEL_DIR="model"
MODEL_FILE="Qwen2.5-3B-Instruct-Q4_K_M.gguf"
MODEL_URL="https://huggingface.co/bartowski/Qwen2.5-3B-Instruct-GGUF/resolve/main/${MODEL_FILE}"
DONE_MARKER="${MODEL_DIR}/.${MODEL_FILE}.complete"

mkdir -p "${MODEL_DIR}"

if [ -f "${DONE_MARKER}" ] && [ -f "${MODEL_DIR}/${MODEL_FILE}" ]; then
  echo "Model already fully downloaded at ${MODEL_DIR}/${MODEL_FILE}, skipping."
  exit 0
fi

echo "Downloading ${MODEL_FILE} (resumes automatically if interrupted)..."
# -C - resumes a partial download instead of restarting from zero -- this
# is the fix for dropped connections on slow/unstable networks.
curl -L -C - --retry 5 --retry-delay 5 -o "${MODEL_DIR}/${MODEL_FILE}" "${MODEL_URL}"
CURL_EXIT=$?

if [ ${CURL_EXIT} -ne 0 ]; then
  echo "ERROR: download did not complete (curl exit code ${CURL_EXIT})." >&2
  echo "Just rerun this script -- it will resume from where it left off." >&2
  exit 1
fi

if [ ! -s "${MODEL_DIR}/${MODEL_FILE}" ]; then
  echo "ERROR: downloaded file is empty or missing." >&2
  exit 1
fi

# Only create the completion marker after curl reports full success --
# this is what prevents a partial file from being silently treated as done.
touch "${DONE_MARKER}"
echo "Done. Model saved to ${MODEL_DIR}/${MODEL_FILE}"
