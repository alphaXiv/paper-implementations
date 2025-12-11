#!/bin/bash

set -e

VERSION=$(python -c 'import olmocr.version; print(olmocr.version.VERSION)')
echo "$VERSION"

docker pull alleninstituteforai/olmocr:v$VERSION
beaker image create --workspace ai2/oe-data-pdf --name olmocr-inference-$VERSION alleninstituteforai/olmocr:v$VERSION
