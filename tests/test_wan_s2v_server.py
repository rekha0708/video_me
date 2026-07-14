from types import SimpleNamespace

import services.wan_s2v_server as server


class _FakeParameter:
    device = "cpu"


class _FakeAudioModel:
    def __init__(self) -> None:
        self.parameter = _FakeParameter()
        self.eval_called = False
        self.requires_grad = True

    def eval(self):
        self.eval_called = True
        return self

    def requires_grad_(self, enabled: bool):
        self.requires_grad = enabled
        return self

    def to(self, device):
        self.parameter.device = str(device)
        return self

    def parameters(self):
        yield self.parameter


def test_place_audio_encoder_on_cuda(monkeypatch) -> None:
    model = _FakeAudioModel()
    pipe = SimpleNamespace(
        device="cuda:0",
        audio_encoder=SimpleNamespace(model=model),
    )
    monkeypatch.setattr(server, "WAN_S2V_AUDIO_ENCODER_DEVICE", "cuda")

    assert server._place_audio_encoder(pipe) == "cuda:0"
    assert model.eval_called is True
    assert model.requires_grad is False


def test_place_audio_encoder_on_cpu(monkeypatch) -> None:
    model = _FakeAudioModel()
    pipe = SimpleNamespace(
        device="cuda:0",
        audio_encoder=SimpleNamespace(model=model),
    )
    monkeypatch.setattr(server, "WAN_S2V_AUDIO_ENCODER_DEVICE", "cpu")

    assert server._place_audio_encoder(pipe) == "cpu"
