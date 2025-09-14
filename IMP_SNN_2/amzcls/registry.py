# Copyright (c) AmazingLab. All rights reserved.

from mmpretrain.registry import DATASETS as MMPRETRAIN_DATASETS
from mmpretrain.registry import DATA_SAMPLERS as MMPRETRAIN_DATA_SAMPLERS
from mmpretrain.registry import EVALUATORS as MMPRETRAIN_EVALUATOR
from mmpretrain.registry import HOOKS as MMPRETRAIN_HOOKS
from mmpretrain.registry import LOG_PROCESSORS as MMPRETRAIN_LOG_PROCESSORS
from mmpretrain.registry import LOOPS as MMPRETRAIN_LOOPS
from mmpretrain.registry import METRICS as MMPRETRAIN_METRICS
from mmpretrain.registry import MODELS as MMPRETRAIN_MODELS
from mmpretrain.registry import MODEL_WRAPPERS as MMPRETRAIN_MODEL_WRAPPERS
from mmpretrain.registry import OPTIMIZERS as MMPRETRAIN_OPTIMIZERS
from mmpretrain.registry import OPTIM_WRAPPERS as MMPRETRAIN_OPTIM_WRAPPERS
from mmpretrain.registry import OPTIM_WRAPPER_CONSTRUCTORS as MMPRETRAIN_OPTIM_WRAPPER_CONSTRUCTORS
from mmpretrain.registry import PARAM_SCHEDULERS as MMPRETRAIN_PARAM_SCHEDULERS
from mmpretrain.registry import RUNNERS as MMPRETRAIN_RUNNERS
from mmpretrain.registry import RUNNER_CONSTRUCTORS as MMPRETRAIN_RUNNER_CONSTRUCTORS
from mmpretrain.registry import Registry
from mmpretrain.registry import TASK_UTILS as MMPRETRAIN_TASK_UTILS
from mmpretrain.registry import TRANSFORMS as MMPRETRAIN_TRANSFORMS
from mmpretrain.registry import VISBACKENDS as MMPRETRAIN_VISBACKENDS
from mmpretrain.registry import VISUALIZERS as MMPRETRAIN_VISUALIZERS
from mmpretrain.registry import WEIGHT_INITIALIZERS as MMPRETRAIN_WEIGHT_INITIALIZERS

__all__ = [
    'RUNNERS', 'RUNNER_CONSTRUCTORS', 'LOOPS', 'HOOKS', 'LOG_PROCESSORS',
    'OPTIMIZERS', 'OPTIM_WRAPPERS', 'OPTIM_WRAPPER_CONSTRUCTORS',
    'PARAM_SCHEDULERS', 'DATASETS', 'DATA_SAMPLERS', 'TRANSFORMS', 'MODELS',
    'MODEL_WRAPPERS', 'WEIGHT_INITIALIZERS', 'BATCH_AUGMENTS', 'TASK_UTILS',
    'METRICS', 'EVALUATORS', 'VISUALIZERS', 'VISBACKENDS'
]

#######################################################################
#                            amzcls.engine                            #
#######################################################################

# Runners like `EpochBasedRunner` and `IterBasedRunner`
RUNNERS = Registry(
    'runner',
    parent=MMPRETRAIN_RUNNERS,
    locations=['amzcls.engine'],
)
# Runner constructors that define how to initialize runners
RUNNER_CONSTRUCTORS = Registry(
    'runner constructor',
    parent=MMPRETRAIN_RUNNER_CONSTRUCTORS,
    locations=['amzcls.engine'],
)
# Loops which define the training or test process, like `EpochBasedTrainLoop`
LOOPS = Registry(
    'loop',
    parent=MMPRETRAIN_LOOPS,
    locations=['amzcls.engine'],
)
# Hooks to add additional functions during running, like `CheckpointHook`
HOOKS = Registry(
    'hook',
    parent=MMPRETRAIN_HOOKS,
    locations=['amzcls.engine'],
)
# Log processors to process the scalar log data.
LOG_PROCESSORS = Registry(
    'log processor',
    parent=MMPRETRAIN_LOG_PROCESSORS,
    locations=['amzcls.engine'],
)
# Optimizers to optimize the model weights, like `SGD` and `Adam`.
OPTIMIZERS = Registry(
    'optimizer',
    parent=MMPRETRAIN_OPTIMIZERS,
    locations=['amzcls.engine'],
)
# Optimizer wrappers to enhance the optimization process.
OPTIM_WRAPPERS = Registry(
    'optimizer_wrapper',
    parent=MMPRETRAIN_OPTIM_WRAPPERS,
    locations=['amzcls.engine'],
)
# Optimizer constructors to customize the hyperparameters of optimizers.
OPTIM_WRAPPER_CONSTRUCTORS = Registry(
    'optimizer wrapper constructor',
    parent=MMPRETRAIN_OPTIM_WRAPPER_CONSTRUCTORS,
    locations=['amzcls.engine'],
)
# Parameter schedulers to dynamically adjust optimization parameters.
PARAM_SCHEDULERS = Registry(
    'parameter scheduler',
    parent=MMPRETRAIN_PARAM_SCHEDULERS,
    locations=['amzcls.engine'],
)

#######################################################################
#                           amzcls.datasets                           #
#######################################################################

# Datasets like `ImageNet` and `CIFAR10`.
DATASETS = Registry(
    'dataset',
    parent=MMPRETRAIN_DATASETS,
    locations=['amzcls.datasets'],
)
# Samplers to sample the dataset.
DATA_SAMPLERS = Registry(
    'data sampler',
    parent=MMPRETRAIN_DATA_SAMPLERS,
    locations=['amzcls.datasets'],
)
# Transforms to process the samples from the dataset.
TRANSFORMS = Registry(
    'transform',
    parent=MMPRETRAIN_TRANSFORMS,
    locations=['amzcls.datasets'],
)

#######################################################################
#                            amzcls.models                            #
#######################################################################

# Neural network modules inheriting `nn.Module`.
MODELS = Registry(
    'model',
    parent=MMPRETRAIN_MODELS,
    locations=['amzcls.models'],
)
# Model wrappers like 'MMDistributedDataParallel'
MODEL_WRAPPERS = Registry(
    'model_wrapper',
    parent=MMPRETRAIN_MODEL_WRAPPERS,
    locations=['amzcls.models'],
)
# Weight initialization methods like uniform, xavier.
WEIGHT_INITIALIZERS = Registry(
    'weight initializer',
    parent=MMPRETRAIN_WEIGHT_INITIALIZERS,
    locations=['amzcls.models'],
)
# Batch augmentations like `Mixup` and `CutMix`.
BATCH_AUGMENTS = Registry(
    'batch augment',
    locations=['amzcls.models'],
)
# Task-specific modules like anchor generators and box coders
TASK_UTILS = Registry(
    'task util',
    parent=MMPRETRAIN_TASK_UTILS,
    locations=['amzcls.models'],
)
# Tokenizer to encode sequence
TOKENIZER = Registry(
    'tokenizer',
    locations=['amzcls.models'],
)

#######################################################################
#                          amzcls.evaluation                          #
#######################################################################

# Metrics to evaluate the model prediction results.
METRICS = Registry(
    'metric',
    parent=MMPRETRAIN_METRICS,
    locations=['amzcls.evaluation'],
)
# Evaluators to define the evaluation process.
EVALUATORS = Registry(
    'evaluator',
    parent=MMPRETRAIN_EVALUATOR,
    locations=['amzcls.evaluation'],
)

#######################################################################
#                         amzcls.visualization                        #
#######################################################################

# Visualizers to display task-specific results.
VISUALIZERS = Registry(
    'visualizer',
    parent=MMPRETRAIN_VISUALIZERS,
    locations=['amzcls.visualization'],
)
# Backends to save the visualization results, like TensorBoard, WandB.
VISBACKENDS = Registry(
    'vis_backend',
    parent=MMPRETRAIN_VISBACKENDS,
    locations=['amzcls.visualization'],
)
