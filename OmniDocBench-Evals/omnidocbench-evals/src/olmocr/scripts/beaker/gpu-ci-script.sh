#/usr/bin/bash

set -ex

git clone https://github.com/allenai/olmocr.git olmocr \
  && cd olmocr \
  && git checkout $GIT_REVISION \
  && /root/.local/bin/uv pip install --system --no-cache \
    .[gpu] \
    pytest \
    --find-links https://flashinfer.ai/whl/cu124/torch2.4/flashinfer/ \
  && bash scripts/run_integration_test.sh



