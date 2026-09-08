# save_models.py
import shutil
import bentoml

with bentoml.models.create(
    name="oil_spill_detector_v1",
    metadata={"mAP": 0.77, "stage": "fallback"},
) as model_ref:
    shutil.copy("models/version_1.pt", model_ref.path_of("model.pt"))
    print(f"Saved: {model_ref}")

with bentoml.models.create(
    name="oil_spill_detector_v2",
    metadata={"mAP": 0.89, "stage": "production"},
) as model_ref:
    shutil.copy("models/version_2.pt", model_ref.path_of("model.pt"))
    print(f"Saved: {model_ref}")