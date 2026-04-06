#!/usr/bin/env bash
set -euo pipefail

ROOT="./datasets"
PARTS=(train val test diffusion)
KEEP_ARCHIVES=0
SKIP_EXISTING=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)
      ROOT="$2"; shift 2 ;;
    --parts)
      shift
      PARTS=()
      while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do
        PARTS+=("$1")
        shift
      done ;;
    --keep-archives)
      KEEP_ARCHIVES=1; shift ;;
    --no-skip-existing)
      SKIP_EXISTING=0; shift ;;
    *)
      echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

mkdir -p "$ROOT"

need_cmd() { command -v "$1" >/dev/null 2>&1 || { echo "missing command: $1" >&2; exit 1; }; }

fetch() {
  local url="$1"
  local out="$2"
  if [[ "$SKIP_EXISTING" -eq 1 && -f "$out" ]]; then
    echo "[skip] $out already exists"
    return 0
  fi
  if command -v aria2c >/dev/null 2>&1; then
    aria2c -x 8 -s 8 -c -o "$(basename "$out")" -d "$(dirname "$out")" "$url"
  else
    wget -c -O "$out" "$url"
  fi
}

cleanup_file() {
  local path="$1"
  if [[ "$KEEP_ARCHIVES" -eq 0 && -f "$path" ]]; then rm -f "$path"; fi
}

cleanup_glob() {
  local pattern="$1"
  if [[ "$KEEP_ARCHIVES" -eq 0 ]]; then rm -f $pattern 2>/dev/null || true; fi
}

download_train() {
  echo "[data] train"
  need_cmd 7z
  need_cmd unzip
  mkdir -p "$ROOT/train"
  pushd "$ROOT/train" >/dev/null
  local base="https://huggingface.co/datasets/sywang/CNNDetection/resolve/main"
  for idx in 001 002 003 004 005 006 007; do
    fetch "${base}/progan_train.7z.${idx}" "progan_train.7z.${idx}"
  done
  if [[ ! -d progan ]]; then
    7z x progan_train.7z.001
    unzip -o progan_train.zip
  fi
  cleanup_glob 'progan_train.7z.*'
  cleanup_file progan_train.zip
  popd >/dev/null
}

download_val() {
  echo "[data] val"
  need_cmd unzip
  mkdir -p "$ROOT/val"
  pushd "$ROOT/val" >/dev/null
  fetch "https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/progan_val.zip" "progan_val.zip"
  if [[ ! -d progan ]]; then unzip -o progan_val.zip; fi
  cleanup_file progan_val.zip
  popd >/dev/null
}

download_test() {
  echo "[data] test"
  need_cmd unzip
  mkdir -p "$ROOT/test"
  pushd "$ROOT/test" >/dev/null
  fetch "https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/CNN_synth_testset.zip" "CNN_synth_testset.zip"
  if [[ ! -d progan ]]; then unzip -o CNN_synth_testset.zip; fi
  cleanup_file CNN_synth_testset.zip
  popd >/dev/null
}

download_diffusion() {
  echo "[data] diffusion"
  python - <<'PY'
import importlib.util, subprocess, sys
if importlib.util.find_spec('gdown') is None:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'gdown>=5.2.0'])
PY
  need_cmd unzip
  mkdir -p "$ROOT/diffusion_datasets"
  pushd "$ROOT/diffusion_datasets" >/dev/null
  if [[ "$SKIP_EXISTING" -eq 1 && -d laion ]]; then
    echo "[skip] diffusion_datasets already extracted"
  else
    python -m gdown "https://drive.google.com/uc?id=1FXlGIRh_Ud3cScMgSVDbEWmPDmjcrm1t" -O diffusion_datasets.zip --fuzzy
    unzip -o diffusion_datasets.zip
    cleanup_file diffusion_datasets.zip
  fi
  popd >/dev/null
}

for part in "${PARTS[@]}"; do
  case "$part" in
    train) download_train ;;
    val) download_val ;;
    test) download_test ;;
    diffusion) download_diffusion ;;
    *) echo "unsupported part: $part" >&2; exit 1 ;;
  esac
done

echo "[data] done"
