set +e
pip install -q pytest
mkdir -p /tmp/dt && cd /tmp/dt
SITE=$(python -c "import ultralytics,os;print(os.path.dirname(ultralytics.__file__))")
python -m pytest -p no:cacheprovider --doctest-modules -q --tb=short -o addopts="" \
  $SITE/data/converter.py $SITE/trackers/utils $SITE/nn/modules/block.py $SITE/data/augment.py $SITE/utils/torch_utils.py $SITE/cfg/__init__.py \
  -k "coco80_to_coco91 or embedding_distance or stracks or C2fPSA or Compose or LetterBox or RandomFlip or Format or RandomPerspective or classify_transforms or classify_augmentations or select_device or cfg2dict or get_cfg or CopyPaste or CutMix or MixUp or Mosaic" 2>&1 | grep -v -E "^\s*$" | head -400
