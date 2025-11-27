"""Small, explicit adapters for the two output projections of Auto-AVSR."""
from pathlib import Path
import torch

BASE_SHA256 = "fbf7cd70ff1c0e694b3030fb779dbb4570f04e4b841d62f9296c229e94878ddb"
HEAD_NAMES = ("ctc.ctc_lo.weight", "ctc.ctc_lo.bias",
              "decoder.output_layer.weight", "decoder.output_layer.bias")


def head_state(model):
    state = model.state_dict()
    return {key: state[key].detach().cpu().clone() for key in HEAD_NAMES}


def set_heads(model, state):
    if set(state) != set(HEAD_NAMES):
        raise ValueError("Adapter must contain exactly the CTC and attention output layers.")
    current = model.state_dict()
    for key, value in state.items():
        if value.shape != current[key].shape or not torch.isfinite(value).all():
            raise ValueError(f"Invalid adapter tensor: {key}")
    with torch.no_grad():
        for key, value in state.items():
            current[key].copy_(value)


def save_adapter(model, destination, metadata):
    destination = Path(destination)
    temporary = destination.with_suffix(".tmp")
    torch.save({"format": 1, "base_sha256": BASE_SHA256,
                "state_dict": head_state(model), "metadata": metadata}, temporary)
    temporary.replace(destination)


def load_adapter(model, source):
    bundle = torch.load(source, map_location="cpu", weights_only=True)
    if bundle.get("format") != 1 or bundle.get("base_sha256") != BASE_SHA256:
        raise ValueError("Adapter is incompatible with the packaged base checkpoint.")
    set_heads(model, bundle["state_dict"])
    return bundle["metadata"]
