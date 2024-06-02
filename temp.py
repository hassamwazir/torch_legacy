import h5py

file_path = "/scratch/whk240/llm/torch_legacy/oxe_torch/episodes/pick_coke_can_place_left_of_spoon.hdf5"

with h5py.File(file_path, "r") as f:
    def print_structure(name, obj):
        if isinstance(obj, h5py.Group):
            print(f"{name} (Group)")
        else:
            print(f"{name} (Dataset)")

    f.visititems(print_structure)
