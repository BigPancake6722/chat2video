def merge_hparams(args, config, is_training=False):
    only_trains = config.get("OnlyTrainParams", [])
    params = ["OptimizationParams", "ModelHiddenParams", "ModelParams", "PipelineParams"]
    for param in params:
        if param in config.keys():
            for key, value in config[param].items():
                if key in only_trains and key in args and not is_training:
                    continue
                if hasattr(args, key):
                    setattr(args, key, value)
    return args


def load_from_file(filename):
    variables = {}
    with open(filename, 'r') as f:
        strs = f.read()
    exec(strs, variables)
    return {k:variables[k] for k in variables if not k.startswith("__")}