import bentoml
for tag in ["oil_spill_detector:vkhhhjemtwkexqfi", "oil_spill_detector:vg7palmmtwa75qfi"]:
    info = bentoml.models.get(tag)
    print(tag, "->", info.info.metadata)