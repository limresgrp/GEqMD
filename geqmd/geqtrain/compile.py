from typing import Callable
import torch
from geqtrain.scripts.evaluate import load_model
from geqtrain.utils._global_options import _set_global_options
from e3nn.util.jit import script
from geqmd.module import ForceModule

from einops._torch_specific import allow_ops_in_compiled_graph  # requires einops>=0.6.1
allow_ops_in_compiled_graph() # needed to compile einops. See https://github.com/arogozhnikov/einops/wiki/Using-torch.compile-with-einops
from openmm.app import *
from openmm.unit.quantity import Quantity
from openmm.unit import unit


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
    field: str,
    topology: Topology,
    positions: Quantity,
    model_position_unit: unit,
    model_energy_unit: unit,
    cutoff: float = None,
    wrapper: Callable = ForceModule,
    device="cpu",
    **kwargs
):
    # -- load GEqTrain model --
    try:
        geqtrain_model, config = load_model(model_filename, device=device)
        _set_global_options(config)
    except:
        geqtrain_model = torch.jit.load(model_filename, map_location=device)

    # -- wrap GEqTrain model for running with OpenMM
    model: ForceModule = wrapper(
        model=geqtrain_model,
        field=field,
        cutoff=cutoff,
        position_unit=model_position_unit,
        energy_unit=model_energy_unit
    )
    model.initialize(topology=topology, device=device, **kwargs)

    # -- compile --
    print("Compiling model...")
    positions = torch.tensor(positions.value_in_unit(model_position_unit), device=device)
    model = _compile_for_deploy(model, example_input=positions)
    print("Compiled & optimized model.")

    # # Compute target cs
    # print("Computing target cs...")
    # geqtrain_model.to(device)
    # input_dict = model.prepare_input_dict(positions)
    # out = model.model(input_dict)

    # # Calculate predicted value as weighted average of predicted bins
    # min_value = 0
    # max_value = 30
    # bins = 128
    # bin_width = (max_value - min_value) / bins
    # bin_centers = bin_width * (torch.arange(bins, device=positions.device) + 0.5)
    # model.target_cs = (torch.softmax(out['node_output'][model.center_atoms], dim=-1) * bin_centers).sum(dim=-1)

    # model.target_cs = out['node_output'][model.center_atoms, 0]
    # print(model.target_cs)

    return model.to(device)