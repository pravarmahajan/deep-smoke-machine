import sys
from i3d_learner import I3dLearner
from ts_learner import TsLearner
from svm_learner import SvmLearner
from lstm_learner import LSTMLearner
try:
    from pt_ts_learner import PtTsLearner
except ImportError:
    PtTsLearner = None
try:
    from fusion_learner import FuseLearner
except ImportError:
    FuseLearner = None
try:
    from late_fusion import LateFusion
except ImportError:
    LateFusion = None


# Train the model
def main(argv):
    if len(argv) < 2:
        print("Usage: python train.py [method]")
        print("Optional usage: python train.py [method] [model_path]")
        return
    method = argv[1]
    if method is None:
        print("Usage: python train.py [method]")
        print("Optional usage: python train.py [method] [model_path]")
        return
    model_path = None
    if len(argv) > 2:
        model_path = argv[2]
    train(method=method, model_path=model_path)


def train(method=None, model_path=None):
    if method == "i3d-rgb":
        if model_path is None:
            model_path = "../data/pretrained_models/i3d_rgb_imagenet_kinetics.pt"
        model = I3dLearner(mode="rgb")
        model.fit(p_model=model_path)
    elif method == "i3d-flow":
        if model_path is None:
            model_path = "../data/pretrained_models/i3d_flow_imagenet_kinetics.pt"
        model = I3dLearner(mode="flow")
        model.fit(p_model=model_path)
    elif method == "i3d-rgb-cv":
        if model_path is None:
            model_path = "../data/pretrained_models/i3d_rgb_imagenet_kinetics.pt"
        i3d_cv("rgb", model_path)
    elif method == "i3d-flow-cv":
        if model_path is None:
            model_path = "../data/pretrained_models/i3d_flow_imagenet_kinetics.pt"
        i3d_cv("flow", model_path)
    elif method == "ts-rgb":
        model = TsLearner(mode="rgb")
        model.fit()
    elif method == "ts-flow":
        model = TsLearner(mode="flow")
        model.fit()
    elif method == "pt-flow":
        model = PtTsLearner()
        model.fit(mode="pt-flow")
    elif method == "svm-rgb":
        model = SvmLearner(mode="rgb")
        model.fit()
    elif method == "svm-flow":
        model = SvmLearner(mode="flow")
        model.fit()
    elif method == "lstm":
        model = LSTMLearner()
        model.fit()
    elif method == "fuse":
        model = FuseLearner()
        model.fit()
    else:
        print("Method not allowed")
        return

# Cross validation of i3d model
def i3d_cv(mode, model_path):
    # Cross validation on the 1st split by camera)
    model = I3dLearner(mode=mode,
            p_metadata_train="../data/split/metadata_train_split_0_by_camera.json",
            p_metadata_validation="../data/split/metadata_validation_split_0_by_camera.json",
            p_metadata_test="../data/split/metadata_test_split_0_by_camera.json")
    model.fit(p_model=model_path)
    # Cross validation on the 2nd split by camera)
    model = I3dLearner(mode=mode,
            p_metadata_train="../data/split/metadata_train_split_1_by_camera.json",
            p_metadata_validation="../data/split/metadata_validation_split_1_by_camera.json",
            p_metadata_test="../data/split/metadata_test_split_1_by_camera.json")
    model.fit(p_model=model_path)
    # Cross validation on the 3rd split by camera)
    model = I3dLearner(mode=mode,
            p_metadata_train="../data/split/metadata_train_split_2_by_camera.json",
            p_metadata_validation="../data/split/metadata_validation_split_2_by_camera.json",
            p_metadata_test="../data/split/metadata_test_split_2_by_camera.json")
    model.fit(p_model=model_path)
    # Cross validation on the split by date
    model = I3dLearner(mode=mode,
            p_metadata_train="../data/split/metadata_train_split_by_date.json",
            p_metadata_validation="../data/split/metadata_validation_split_by_date.json",
            p_metadata_test="../data/split/metadata_test_split_by_date.json")
    model.fit(p_model=model_path)

if __name__ == "__main__":
    main(sys.argv)
