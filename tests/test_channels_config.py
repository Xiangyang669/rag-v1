"""图片通道的配置项。

引擎与阈值都必须可配：换 OCR / 换多模态模型不该改代码，而阈值是
「文字为主 vs 视觉为主」的分流依据，不同语料下取值会变。
"""

from ragv1 import config
from ragv1.ingest.image_channels import ChannelConfig


def test_config_defaults_match_spec():
    assert config.OCR_ENGINE == "rapidocr"
    assert config.VLM_ENGINE == "siliconflow"
    assert config.VLM_MODEL == "Qwen/Qwen2.5-VL-32B-Instruct"
    assert config.OCR_MIN_CHARS == 10
    assert config.OCR_MIN_CONF == 0.5
    assert config.ENABLE_IMAGE is True
    assert config.IMAGE_CACHE_DIR.name == "images"


def test_cache_dir_lives_under_index_dir():
    assert config.IMAGE_CACHE_DIR.parent == config.INDEX_DIR


def test_channel_config_takes_defaults_from_config():
    c = ChannelConfig()
    assert c.min_chars == config.OCR_MIN_CHARS
    assert c.min_conf == config.OCR_MIN_CONF
    assert c.enabled == config.ENABLE_IMAGE


def test_engine_settings_are_env_overridable(monkeypatch):
    """可配置是需求：换引擎不该改代码"""
    monkeypatch.setenv("OCR_ENGINE", "none")
    import importlib

    importlib.reload(config)
    assert config.OCR_ENGINE == "none"

    monkeypatch.delenv("OCR_ENGINE")
    importlib.reload(config)
    assert config.OCR_ENGINE == "rapidocr"
