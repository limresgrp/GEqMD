import os
import torch
from geqtrain.train import Trainer
from geqtrain.utils._global_options import _set_global_options
from e3nn.util.jit import script

from einops._torch_specific import allow_ops_in_compiled_graph  # requires einops>=0.6.1
allow_ops_in_compiled_graph() # needed to compile einops. See https://github.com/arogozhnikov/einops/wiki/Using-torch.compile-with-einops
from openmm.app import *


def _compile_for_deploy(model, example_input):
    model.eval()

    try:
        if not isinstance(model, torch.jit.ScriptModule):
            model = script(model)
    except Exception as e:
        print(f"Error during scripting the model: {e}")
        print("Trying to trace the model instead.")
        try:
            model = torch.jit.trace(model, example_input, strict=False)
        except Exception as trace_e:
            print(f"Error during tracing the model: {trace_e}")
            raise trace_e

    return model

def load_and_init_model(
    model_filename: str,
    wrapper: torch.nn.Module,
    positions: torch.Tensor,
    device="cpu",
    **kwargs
):
    # -- load GEqTrain model --
    print("Loading model from training...")
    geqtrain_model, config = Trainer.load_model_from_training_session(
        os.path.dirname(model_filename),
        os.path.basename(model_filename),
        for_inference=True,
        device=device)
    _set_global_options(config)

    # -- wrap GEqTrain model for running with OpenMM
    model = wrapper(model=geqtrain_model)
    model.initialize(device=device, positions=positions, **kwargs)

    # -- compile --
    print("Compiling model...")
    positions = positions.to(device)
    model = _compile_for_deploy(model, example_input=positions)
    print("Compiled & optimized model.")

    # Compute target cs
    print("Computing target cs...")
    geqtrain_model.to(device)
    input_dict = model.prepare_input_dict(positions)
    out = model.model(input_dict)
    model.target_cs = out['node_output'][model.filter_atoms, 0]
    print("Complete!")

    return model