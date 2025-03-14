import re
from typing import Dict

# This is a weird hack to avoid Intel MKL issues on the cluster when this is called as a subprocess of a process that has itself initialized PyTorch.
# Since numpy gets imported later anyway for dataset stuff, this shouldn't affect performance.
import numpy as np  # noqa: F401
import torch
from e3nn.util.jit import compile_mode

from einops._torch_specific import allow_ops_in_compiled_graph  # requires einops>=0.6.1
allow_ops_in_compiled_graph() # needed to compile einops. See https://github.com/arogozhnikov/einops/wiki/Using-torch.compile-with-einops


@compile_mode("script")
class ForceModule(torch.nn.Module):

    def __init__(self, model, recompute_steps=100, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.recompute_steps = recompute_steps
        self.register_buffer('counter', torch.tensor(0.))
    
    def initialize(self, topology, atom_type_config_file, atom_type_map, positions, atom_names_filters, device='cpu', **kwargs):
        self.device = device

        # Initialize the NMRDataset with the config file
        from csnet.utils.dataset import AtomTypeAssigner
        atom_type_assigner = AtomTypeAssigner(atom_type_config_file, atom_type_map=atom_type_map)

        atom_data, atom_numbers, atom_names = [], [], []
        for a in topology.atoms():
            resname  = a.residue.name
            atomname = a.name
            element  = a.element._symbol
            atom_data.append({
                'resname': resname,
                'atomname': atomname,
                'element': element,
            })
            atom_numbers.append(a.element._atomic_number)
            atom_names.append(atomname)
        atom_types = atom_type_assigner.assign_atom_types(atom_data)

        node_types    = torch.tensor(atom_types['atom_types'])
        atom_restypes = torch.tensor(atom_types['atom_restypes'])
        atom_numbers  = torch.tensor(atom_numbers)
        batch         = torch.zeros_like(atom_numbers)
        
        self.atom_names = atom_names
        filter_atoms = torch.zeros_like(atom_numbers).bool()

        if atom_names_filters is None:
            filter_atoms[:] = True
        else:
            for i, atom_name in enumerate(atom_names):
                for pattern in atom_names_filters:
                    if re.match(pattern, atom_name):
                        filter_atoms[i] = True
                        break
        
        target_cs  = torch.zeros((filter_atoms.sum(), 1), dtype=torch.float32)
        
        self.register_buffer('node_types',    node_types)
        self.register_buffer('atom_restypes', atom_restypes)
        self.register_buffer('atom_numbers',  atom_numbers)
        self.register_buffer('batch',         batch)
        self.register_buffer('filter_atoms',  filter_atoms)
        self.register_buffer('target_cs',     target_cs)

        self.to(device)
        self.update_edge_index(positions.to(device))

    def update_edge_index(self, positions: torch.Tensor):
        # Compute pairwise distances using torch
        dist_matrix = torch.cdist(positions[self.filter_atoms], positions)
        
        # Create a mask for distances within the threshold
        mask = (dist_matrix > 0.01) & (dist_matrix <= 0.5)
        
        # Get the indices of the source and target nodes
        src_index, trg_index = torch.where(mask)
        
        # Filter the indices based on the node center mask
        node_center_mask = self.filter_atoms.nonzero().flatten()
        self.edge_index = torch.stack([node_center_mask[src_index], trg_index], dim=0).to(positions.device)

    def update_edge_index_if_needed(self, positions):
        if torch.remainder(self.counter, self.recompute_steps) == 0:
            self.update_edge_index(positions)
        self.counter += 1
    
    def prepare_input_dict(self, positions) -> Dict[str, torch.Tensor]:
        return {
            'pos':           positions.float() * 10.,
            'batch':         self.batch,
            'edge_index':    self.edge_index,
            'node_types':    self.node_types,
            'atom_restypes': self.atom_restypes,
            'atom_numbers':  self.atom_numbers,
        }

    def forward(self, positions):
        """The forward method returns the energy computed from positions.

        Parameters
        ----------
        positions : torch.Tensor with shape (nparticles,3)
           positions[i,k] is the position (in nanometers) of spatial dimension k of particle i

        Returns
        -------
        potential : torch.Scalar
           The potential energy (in kJ/mol)
        """
        self.update_edge_index_if_needed(positions)

        input_dict = self.prepare_input_dict(positions)
        out = self.model(input_dict)
        
        pred_cs = out['node_output']
        energy = torch.sum((pred_cs[self.filter_atoms, 0] - self.target_cs)**2)

        # center_pos = out['pos'][out['edge_index'][0]]
        # energy = torch.cdist(center_pos, center_pos).sum() * 1.e-4
        
        return energy