from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.modeling_utils import AttentionInterface

from ..ops.attention import ALL_FLASH_ATTENTION_FUNCTIONS, flash_attention_forward
from ..utils import logging


logger = logging.get_logger(__name__)

# Global AttentionInterface shared by all models which do not need to overwrite any of the existing ones
ALL_ATTENTION_FUNCTIONS: AttentionInterface = AttentionInterface()

ALL_ATTENTION_FUNCTIONS["flash_attention_3"] = flash_attention_forward
ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = flash_attention_forward

ALL_FLASH_ATTENTION_FUNCTIONS["flash_attention_2"] = _flash_attention_forward
ALL_FLASH_ATTENTION_FUNCTIONS["flash_attention_3"] = _flash_attention_forward

logger.warning_once("✅ Transformers ALL_ATTENTION_FUNCTIONS patched with new flash_attention_forward in VeOmni")
