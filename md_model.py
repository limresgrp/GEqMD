import argparse
import numpy as np  # noqa: F401
import torch
from einops._torch_specific import allow_ops_in_compiled_graph  # requires einops>=0.6.1
from openmmtorch import TorchForce
from openmm.app import PDBFile, ForceField, Simulation, Modeller, PDBReporter, StateDataReporter
from openmm import Platform, LangevinIntegrator
from openmm.unit import kelvin, picosecond, femtosecond, nanometers
from geqmd.geqtrain.compile import load_and_init_model
from geqmd.module import ForceModule
import sys

allow_ops_in_compiled_graph()  # needed to compile einops. See https://github.com/arogozhnikov/einops/wiki/Using-torch.compile-with-einops

def main(model_filename, input_file, target_input_file, atom_names_filters, output_filename, num_simulations, num_steps):
    config_file = "/storage_common/angiod/nmr/data/atom_type_config.protein.yaml"
    atom_type_map = "/storage_common/angiod/nmr/bmrb/atom_type_map.yaml"

    forcefield = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')

    # load input molecule and initialize model
    target_input = read_inpout_file(target_input_file, forcefield)

    model = load_and_init_model(
        model_filename,
        ForceModule,
        positions=torch.tensor(target_input.positions.value_in_unit(nanometers)),
        topology=target_input.topology,
        atom_type_config_file=config_file,
        atom_type_map=atom_type_map,
        atom_names_filters=atom_names_filters,
        device='cuda',
    )

    torch.jit.save(model, 'model.pt', _extra_files={})

    print("Reading input file")
    input = read_inpout_file(input_file, forcefield)
    print("Starting MD simulations")
    for i in range(1, num_simulations + 1):
        print(f"Simulation ID: {i}")
        system = forcefield.createSystem(input.topology)

        # Construct using a serialized module
        torch_force = TorchForce(model)

        # Add the TorchForce to your System
        system.addForce(torch_force)

        # Create an integrator with a time step of 1 fs
        temperature = 310 * kelvin # 298.15 * kelvin
        frictionCoeff = 1.0 / picosecond # .1 / picosecond
        timeStep = 2 * femtosecond
        integrator = LangevinIntegrator(temperature, frictionCoeff, timeStep)
        # integrator = VerletIntegrator(timeStep)

        # Create a simulation and set the initial positions and velocities
        platform = Platform.getPlatformByName('CUDA') #'CUDA'
        simulation = Simulation(input.topology, system, integrator, platform=platform)
        simulation.context.setPositions(input.positions)

        output_file_with_index = f'{output_filename}.{i}.pdb'
        simulation.reporters.append(PDBReporter(output_file_with_index, 100))
        simulation.reporters.append(StateDataReporter(sys.stdout, 100, step=True, potentialEnergy=True, temperature=True))

        simulation.step(num_steps)

def read_inpout_file(input_file, forcefield):
    assert input_file.endswith('.pdb')
    input = PDBFile(input_file)

    # Create a Modeller object
    modeller = Modeller(input.topology, input.positions)
    modeller.addExtraParticles(forcefield)

    input.topology = modeller.topology
    input.positions = modeller.positions
    return input

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run molecular dynamics simulation with a given model and input file.')
    parser.add_argument('model_filename', type=str, help='Path to the model file')
    parser.add_argument('input_file', type=str, help='Path to the input PDB file')
    parser.add_argument('target_input_file', type=str, help='Path to the target input PDB file')
    parser.add_argument('-o', '--output_filename', type=str, default='output', help='Base name for the output PDB file (default: output)')
    parser.add_argument('-f', '--atom_names_filters', nargs='*', default=None, help='List of atom names to filter (default: None)')
    parser.add_argument('-n', '--num_simulations', type=int, default=10, help='Number of simulations to run (default: 10)')
    parser.add_argument('-s', '--num_steps', type=int, default=100000, help='Number of steps per simulation (default: 100000)')
    args = parser.parse_args()
    main(args.model_filename, args.input_file, args.target_input_file, args.atom_names_filters, args.output_filename, args.num_simulations, args.num_steps)
