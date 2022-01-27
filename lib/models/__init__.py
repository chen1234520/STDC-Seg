
import imp
import sys
from .bisenetv1 import BiSeNetV1
from .bisenetv2 import BiSeNetV2
from .model_stages import BiSeNet   # STDC-seg
from .model_stages_modifies import BiSeNet_cutNet, BiSeNet_cutNet_infer # STDC-seg
from .model_stages_DW import BiSeNet_DW   # STDC-seg
from .model_stages_del import BiSeNet_delNet, BiSeNet_delNet_infer

model_factory = {
    'bisenetv1': BiSeNetV1,
    'bisenetv2': BiSeNetV2,
    'STDC': BiSeNet,
    'STDC_Cut':BiSeNet_cutNet,
    'STDC_DW':BiSeNet_DW,
    'STDC_Cut_infer':BiSeNet_cutNet_infer,
    'STDC_Del_infer':BiSeNet_delNet_infer,
    'STDC_Del':BiSeNet_delNet
}
