import pytest

pytest.importorskip("tensorflow")

from server.model_service import count_params, find_layer, load_model  # noqa: E402


@pytest.fixture(scope="module")
def model():
    return load_model()


def test_model_accepts_224_rgb_and_outputs_four_classes(model):
    assert model.input_shape[-3:] == (224, 224, 3)
    assert model.output_shape[-1] == 4


def test_model_has_heatmap_layer(model):
    assert find_layer(model, "block5_conv3") is not None


def test_model_is_the_block4_block5_variant(model):
    assert find_layer(model, "block4_conv1").trainable is True
    assert find_layer(model, "block5_conv3").trainable is True
    assert find_layer(model, "block3_conv1").trainable is False
    assert find_layer(model, "block1_conv1").trainable is False


def test_parameter_counts_match_summary_csv(model):
    assert count_params(model, trainable_only=False) == 14_716_740
    assert count_params(model, trainable_only=True) == 12_981_252
