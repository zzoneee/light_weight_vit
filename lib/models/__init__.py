from . import operations
from . import operations_resnet 

# models which use timm's registry
try:
    import timm
    _has_timm = True
except ModuleNotFoundError:
    _has_timm = False

if _has_timm:
    from . import lightvit
    from . import lightvit_dualToken
    from . import lightvit_dualToken_fix
    from . import lightvit_dualToken_conv
    from . import lightvit_dualToken_conv_convMergePatch
    from . import lightvit_dualToken_conv_hydra
    from . import lightvit_dualToken_convEdgenext
    from . import lightvit_dualToken_convEdgenextAll
    from . import lightvit_dualToken_convEdgenext_mix
    from . import lightvit_dualToken_convEdgenext_resGT
    from . import lightvit_dualToken_convEdgenext_normGT
    from . import lightvit_dualToken_convEdgenext_mix_normGT
    from . import lightvit_dualToken_convEdgenext_mix_normGT_wsa
    from . import lightvit_dualToken_convEdgenext_mix_normGT_a03
    from . import lightvit_dualToken_convEdgenext_mix_normGT_ds
    from . import lightvit_dualToken_convEdgenext_mix_normGT_mlp
    from . import lightvit_dualToken_convEdgenext_mix_normGT_gt36
    from . import lightvit_dualToken_convEdgenext_mlp_normGT_gt36
    from . import lightvit_dualToken_convEdgenext_mlp_normGT_gt25
    from . import lightvit_dualToken_convEdgenext_mlp_normGT_gt16
    from . import lightvit_dualToken_convEdgenext_mlp_normGT_GT8
    from . import lightvit_dualToken_convEdgenext_mlp_normGT_nodsGT8
    from . import lightvit_dualToken_convEdgenext_mlp_normGT_gt64
    from . import lightvit_dualToken_convEdgenext_mix_normGT_wsa
    from . import lightvit_dualToken_convEdgenext_mix_normGT_ds
    from . import lightvit_dualToken_convEdgenext_mix_normGT_mlp_nogt
    from . import lightvit_dualToken_convEdgenext_mlp_normGT_gt9

    from . import lightvit_dualToken_convEdgenext_mix_normGT_posGT
    from . import lightvit_dualToken_convEdgenext_mix_normGT_posCosGT
    from . import lightvit_dualToken_convEdgenext_mix_normGT_posCos
