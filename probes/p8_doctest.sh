set +e
pip install -q pytest
git clone -q --depth 1 https://github.com/ultralytics/ultralytics /tmp/ul
cd /tmp/ul
mkdir -p /tmp/dt && cd /tmp/dt
cp /tmp/ul/ultralytics/assets/bus.jpg image.jpg 2>/dev/null
SITE=$(python -c "import ultralytics,os;print(os.path.dirname(ultralytics.__file__))")
echo "SITE=$SITE"
python -m pytest -p no:cacheprovider --doctest-modules -q --tb=line -rf --continue-on-collection-errors -o addopts="" \
  $SITE/utils/ops.py $SITE/utils/metrics.py $SITE/utils/instance.py $SITE/utils/__init__.py $SITE/utils/checks.py $SITE/utils/files.py \
  $SITE/utils/plotting.py $SITE/engine/results.py $SITE/data/utils.py $SITE/data/converter.py $SITE/cfg/__init__.py $SITE/utils/nms.py \
  $SITE/utils/tal.py $SITE/utils/torch_utils.py $SITE/trackers/utils $SITE/data/augment.py $SITE/utils/downloads.py $SITE/data/split.py \
  $SITE/nn/modules $SITE/utils/loss.py $SITE/data/loaders.py $SITE/data/build.py $SITE/engine/model.py 2>&1 | tail -150
