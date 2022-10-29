
from .transformer_layer_adapter import (
    TransformerEncoderLayerAdapter,
    TransformerDecoderLayerAdapter,
)
from .transformer_decoder_adapter import TransformerDecoderAdapter
from .transformer_encoder_adapter import TransformerEncoderAdapter
from .transformer_adapter import(
    TransformerWithAdapter,
    base_architecture,
)
from .label_smoothed_cross_entropy_adapter import LabelSmoothedCrossEntropyCriterionWithAdapter
from .translation_multi_simple_epoch_adapter import TranslationMultiSimpleEpochTaskWithAdapter
from .sequence_generator_adapter import (
    SequenceGeneratorAdapter,
    EnsembleModelAdapter,
)
from .trainer_control import Trainer_con
