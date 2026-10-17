#!/bin/bash
# Downloads CIFAR-100 (ID) and the five CIFAR OOD test sets into <repo>/data, in the layout the scripts expect:
#   data/cifarpy/cifar-100-python/      (torchvision CIFAR100 root = data/cifarpy)
#   data/svhn/{train,test,extra}_32x32.mat
#   data/places365_test/unknown/*.jpg   (full Places365 test_256 set, 328,500 images; a 10k subset is drawn at test time)
#   data/LSUN_C/unknown/*.jpg           (LSUN crop, 10,000 images, from the ODIN release)
#   data/iSUN/unknown/*.jpg             (8,925 images, from the ODIN release)
#   data/dtd/images/<category>/*.jpg    (Describable Textures, 5,640 images)
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
DATA=${DATA_DIR:-$REPO/data}
ARCH=$DATA/_downloads
mkdir -p "$ARCH" "$DATA/cifarpy" "$DATA/svhn" "$DATA/places365_test/unknown" "$DATA/LSUN_C/unknown" "$DATA/iSUN/unknown"

echo "[cifar100]"; wget -nv -c -O "$ARCH/cifar-100-python.tar.gz" https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
tar -xzf "$ARCH/cifar-100-python.tar.gz" -C "$DATA/cifarpy"
echo "[svhn]"; for f in train_32x32.mat test_32x32.mat extra_32x32.mat; do wget -nv -c -O "$ARCH/$f" "http://ufldl.stanford.edu/housenumbers/$f"; cp -f "$ARCH/$f" "$DATA/svhn/"; done
echo "[dtd]"; wget -nv -c -O "$ARCH/dtd-r1.0.1.tar.gz" https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz
tar -xzf "$ARCH/dtd-r1.0.1.tar.gz" -C "$DATA"
echo "[LSUN_C]"; wget -nv -c -O "$ARCH/LSUN.tar.gz" "https://www.dropbox.com/s/fhtsw1m3qxlwj6h/LSUN.tar.gz?dl=1"
tar -xzf "$ARCH/LSUN.tar.gz" --strip-components=2 --skip-old-files -C "$DATA/LSUN_C/unknown"
echo "[iSUN]"; wget -nv -c -O "$ARCH/iSUN.tar.gz" "https://www.dropbox.com/s/ssz7qxfqae0cca5/iSUN.tar.gz?dl=1"
tar -xzf "$ARCH/iSUN.tar.gz" --strip-components=2 --skip-old-files -C "$DATA/iSUN/unknown"
echo "[places365 test_256, ~20GB]"; wget -nv -c -O "$ARCH/test_256.tar" http://data.csail.mit.edu/places/places365/test_256.tar
tar -xf "$ARCH/test_256.tar" --strip-components=1 -C "$DATA/places365_test/unknown"
echo "[counts]"
for d in cifarpy/cifar-100-python svhn dtd/images LSUN_C/unknown iSUN/unknown places365_test/unknown; do echo "$(find "$DATA/$d" -type f | wc -l) files  $d"; done
echo DONE
