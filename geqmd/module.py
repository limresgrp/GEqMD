import re
from typing import Dict, Optional
from openmm import unit

# This is a weird hack to avoid Intel MKL issues on the cluster when this is called as a subprocess of a process that has itself initialized PyTorch.
# Since numpy gets imported later anyway for dataset stuff, this shouldn't affect performance.
import numpy as np  # noqa: F401
import torch
from e3nn.util.jit import compile_mode

from einops._torch_specific import allow_ops_in_compiled_graph  # requires einops>=0.6.1
allow_ops_in_compiled_graph() # needed to compile einops. See https://github.com/arogozhnikov/einops/wiki/Using-torch.compile-with-einops


@compile_mode("script")
class ForceModule(torch.nn.Module):

    def __init__(
            self,
            model,
            field: str,
            cutoff: Optional[float] = None,
            position_unit = unit.angstrom,
            energy_unit   = unit.kilocalories_per_mole,
            recompute_steps: int = 100,
            **kwargs
        ):
        super().__init__(**kwargs)
        self.model = model
        self.field = field
        self.recompute_steps = recompute_steps
        self.register_buffer('counter', torch.tensor(0, dtype=torch.long))
        self.cutoff = cutoff
        self.position_unit = position_unit
        self.energy_unit = energy_unit
        self.register_buffer('openmm_unit2model_unit_position', torch.tensor((1. * unit.nanometer).value_in_unit(self.position_unit), dtype=torch.float32))
        self.register_buffer('model_unit2openmm_unit_energy',   torch.tensor((1. * unit.kilojoule_per_mole).value_in_unit(self.energy_unit), dtype=torch.float32))
        self.edge_index = torch.tensor([[0,], [0,]])
    
    def initialize(self, topology, device='cpu', atom_names_filters=None, resname_filters=None, neighbor_names_filters=None, neighbor_resname_filters=None, **kwargs):
        self.device = device
        atom_numbers, atom_resnames, atom_names, atom_elements = [], [], [], []
        for atom in topology.atoms():
            atom_numbers.append(atom.element._atomic_number)
            atom_resnames.append(atom.residue.name)
            atom_names.append(atom.name)
            atom_elements.append(atom.element._symbol)
        
        self.initialize_input_buffers(atom_numbers, atom_resnames, atom_names, atom_elements)
        num_atoms = len(atom_numbers)

        # Center atoms mask: filter by atom name or resname
        filtered_center_atoms = torch.zeros((num_atoms), dtype=bool)
        for i, (atom_name, resname) in enumerate(zip(atom_names, atom_resnames)):
            name_match = any(re.match(pat, atom_name) for pat in atom_names_filters or [])
            resname_match = any(re.match(pat, resname) for pat in resname_filters or [])
            if (atom_names_filters is None and resname_filters is None) or name_match or resname_match:
                filtered_center_atoms[i] = True

        # Neighboring atoms mask: filter by atom name or resname
        filtered_neighbour_atoms = torch.zeros((num_atoms), dtype=bool)
        for i, (atom_name, resname) in enumerate(zip(atom_names, atom_resnames)):
            name_match = any(re.match(pat, atom_name) for pat in neighbor_names_filters or [])
            resname_match = any(re.match(pat, resname) for pat in neighbor_resname_filters or [])
            if (neighbor_names_filters is None and neighbor_resname_filters is None) or name_match or resname_match:
                filtered_neighbour_atoms[i] = True
        
        # All atoms mask
        filtered_all_atoms = torch.logical_or(filtered_center_atoms, filtered_neighbour_atoms)
        
        # Batch
        filtered_num_atoms = sum(filtered_all_atoms).long()
        batch              = torch.zeros((filtered_num_atoms), dtype=torch.long)

        self.register_buffer('center_atoms'   , filtered_center_atoms)
        self.register_buffer('neighbour_atoms', filtered_neighbour_atoms)
        self.register_buffer('all_atoms'      , filtered_all_atoms)
        self.register_buffer('batch'          , batch)

        self.to(device)
    
    def initialize_input_buffers(self, atom_numbers, atom_resnames, atom_names, atom_elements):
        self.register_buffer('atom_numbers', torch.tensor(atom_numbers, dtype=torch.long))

    def update_edge_index(self, positions: torch.Tensor):
        # Get positions of all relevant atoms
        all_atom_positions = positions[self.all_atoms]
        # Compute pairwise distances
        dist_matrix = torch.cdist(all_atom_positions, all_atom_positions)
        # Apply cutoff (if self.cutoff is None, use all distances > 0)
        cutoff = self.cutoff if self.cutoff is not None else torch.inf
        mask = (dist_matrix > 0) & (dist_matrix <= cutoff)
        # Indices in all_atoms
        src_index, trg_index = torch.where(mask)

        # Map indices in all_atoms back to global atom indices
        all_atom_indices       = self.all_atoms.nonzero().flatten()
        center_atom_indices    = self.center_atoms.nonzero().flatten()
        neighbour_atom_indices = self.neighbour_atoms.nonzero().flatten()

        # Only keep edges where source is a center atom and target is a neighbour atom
        src_global = all_atom_indices[src_index]
        trg_global = all_atom_indices[trg_index]
        src_mask = torch.isin(src_global, center_atom_indices)
        trg_mask = torch.isin(trg_global, neighbour_atom_indices)
        edge_mask = src_mask & trg_mask

        # Final edge_index: shape [2, num_edges]
        self.edge_index = torch.stack([src_global[edge_mask], trg_global[edge_mask]], dim=0).to(positions.device)

    def update_edge_index_if_needed(self, positions):
        if torch.remainder(self.counter, self.recompute_steps) == 0:
            self.update_edge_index(positions)
        self.counter += 1
    
    def prepare_input_dict(self, positions) -> Dict[str, torch.Tensor]:
        return {
            'pos':           positions[self.all_atoms],
            'batch':         self.batch,
            'edge_index':    self.edge_index,
            'node_types':    self.atom_numbers,
        }

    def forward(self, positions: torch.Tensor):
        positions = self.openmm_unit2model_unit_position * positions
        self.update_edge_index_if_needed(positions)

        input_dict = self.prepare_input_dict(positions)
        energy = self.model(input_dict)[self.field]
        
        return energy * self.model_unit2openmm_unit_energy


# @compile_mode("script")
# class CSModule(torch.nn.Module):

#     def __init__(self, model, recompute_steps=100, **kwargs):
#         super().__init__(**kwargs)
#         self.model = model
#         self.recompute_steps = recompute_steps
#         self.register_buffer('counter', torch.tensor(0.))
    
#     def initialize(self, topology, atom_type_config_file, atom_type_map, positions, atom_names_filters, energy_scale=1., device='cpu', **kwargs):
#         self.device = device

#         # Initialize the NMRDataset with the config file
#         from csnet.utils.dataset import AtomTypeAssigner
#         atom_type_assigner = AtomTypeAssigner(atom_type_config_file, atom_type_map=atom_type_map)

#         atom_data, atom_numbers, atom_names = [], [], []
#         for a in topology.atoms():
#             resname  = a.residue.name
#             atomname = a.name
#             element  = a.element._symbol
#             atom_data.append({
#                 'resname': resname,
#                 'atomname': atomname,
#                 'element': element,
#             })
#             atom_numbers.append(a.element._atomic_number)
#             atom_names.append(atomname)
#         atom_types = atom_type_assigner.assign_atom_types(atom_data)

#         node_types    = torch.tensor(atom_types['atom_types'])
#         atom_restypes = torch.tensor(atom_types['atom_restypes'])
#         atom_numbers  = torch.tensor(atom_numbers)
#         batch         = torch.zeros_like(atom_numbers)
        
#         center_atoms = torch.zeros_like(atom_numbers).bool()
#         if atom_names_filters is None:
#             center_atoms[:] = True
#         else:
#             for i, atom_name in enumerate(atom_names):
#                 for pattern in atom_names_filters:
#                     if re.match(pattern, atom_name):
#                         center_atoms[i] = True
#                         break
#         neighbour_atoms = torch.zeros_like(atom_numbers).bool()
#         for i, atom_restype in enumerate(atom_restypes):
#             if atom_restype not in [-1]:
#                 neighbour_atoms[i] = True
        
#         target_cs    = torch.zeros((center_atoms.sum(), 1), dtype=torch.float32)
#         energy_scale = torch.tensor(energy_scale, dtype=torch.float32)
        
#         self.register_buffer('node_types'        , node_types)
#         self.register_buffer('atom_restypes'     , atom_restypes)
#         self.register_buffer('atom_numbers'      , atom_numbers)
#         self.register_buffer('batch'             , batch)
#         self.register_buffer('center_atoms'      , center_atoms)
#         self.register_buffer('neighbour_atoms', neighbour_atoms)
#         self.register_buffer('target_cs'         , target_cs)
#         self.register_buffer('energy_scale'      , energy_scale)

#         self.to(device)
#         self.update_edge_index(positions.to(device))

#     def update_edge_index(self, positions: torch.Tensor):
#         # Compute pairwise distances using torch
#         dist_matrix = torch.cdist(positions[self.center_atoms], positions[self.neighbour_atoms])
        
#         # Create a mask for distances within the threshold
#         mask = (dist_matrix > 0.01) & (dist_matrix <= 0.5)
        
#         # Get the indices of the source and target nodes
#         src_index, trg_index = torch.where(mask)
        
#         # Filter the indices based on the node center mask
#         node_center_mask = self.center_atoms.nonzero().flatten()
#         self.edge_index = torch.stack([node_center_mask[src_index], trg_index], dim=0).to(positions.device)

#     def update_edge_index_if_needed(self, positions):
#         if torch.remainder(self.counter, self.recompute_steps) == 0:
#             self.update_edge_index(positions)
#         self.counter += 1
    
#     def prepare_input_dict(self, positions) -> Dict[str, torch.Tensor]:
#         return {
#             'pos':           positions[self.neighbour_atoms].float() * 10.,
#             'batch':         self.batch,
#             'edge_index':    self.edge_index,
#             'node_types':    self.atom_numbers,
#             # 'node_types':    self.node_types,
#             # 'atom_restypes': self.atom_restypes,
#             # 'atom_numbers':  self.atom_numbers,
#         }

#     def forward(self, positions):
#         """The forward method returns the energy computed from positions.

#         Parameters
#         ----------
#         positions : torch.Tensor with shape (nparticles,3)
#            positions[i,k] is the position (in nanometers) of spatial dimension k of particle i

#         Returns
#         -------
#         potential : torch.Scalar
#            The potential energy (in kJ/mol)
#         """
#         self.update_edge_index_if_needed(positions)

#         input_dict = self.prepare_input_dict(positions)
#         out = self.model(input_dict)
        
#         pred_cs = out['node_output']
#         energy = self.energy_scale * torch.sum((pred_cs[self.center_atoms, 0] - self.target_cs)**2)

#         # # Calculate predicted value as weighted average of predicted bins
#         # min_value = 0
#         # max_value = 30
#         # bins = 128
#         # bin_width = (max_value - min_value) / bins
#         # bin_centers = bin_width * (torch.arange(bins, device=positions.device) + 0.5)
#         # energy = self.energy_scale * torch.sum(((torch.softmax(pred_cs[self.center_atoms], dim=-1) * bin_centers).sum(dim=-1) - self.target_cs)**2)

#         # Fold as a ball
#         # center_pos = out['pos'][out['edge_index'][0]]
#         # energy = torch.cdist(center_pos, center_pos).sum() * 1.e-4
        
#         return energy