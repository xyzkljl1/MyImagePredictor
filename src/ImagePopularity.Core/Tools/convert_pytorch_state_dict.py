import collections
import io
import json
import os
import pickle
import sys
import zipfile
from math import prod


FORMAT_VERSION = 2
DOTNET_TICKS_AT_UNIX_EPOCH = 621355968000000000


class _TensorSpec:
    def __init__(self, storage, storage_offset, size, stride):
        self.storage = storage
        self.storage_offset = storage_offset
        self.size = list(size)
        self.stride = list(stride)


class _StateDictUnpickler(pickle.Unpickler):
    def persistent_load(self, pid):
        return pid

    def find_class(self, module, name):
        if module == "collections" and name == "OrderedDict":
            return collections.OrderedDict
        if module == "torch._utils" and name in ("_rebuild_tensor_v2", "_rebuild_tensor"):
            return lambda storage, storage_offset, size, stride, *rest: _TensorSpec(storage, storage_offset, size, stride)
        if module == "torch" and name.endswith("Storage"):
            return (module, name)
        raise pickle.UnpicklingError(f"Unsupported class reference: {module}.{name}")


def _dtype_info(storage_type_name: str):
    if storage_type_name == "FloatStorage":
        return "float32", 4
    raise RuntimeError(f"Unsupported storage type: {storage_type_name}")


def _is_contiguous(shape, stride):
    expected = 1
    for dim, step in zip(reversed(shape), reversed(stride)):
        if dim == 1:
            continue
        if step != expected:
            return False
        expected *= dim
    return True


def _load_state_dict(zip_file: zipfile.ZipFile):
    data = zip_file.read("archive/data.pkl")
    return _StateDictUnpickler(io.BytesIO(data)).load()


def _unpack_storage_pid(storage_pid):
    if (
        isinstance(storage_pid, tuple)
        and len(storage_pid) == 5
        and storage_pid[0] == "storage"
    ):
        return storage_pid

    if (
        isinstance(storage_pid, tuple)
        and len(storage_pid) >= 2
        and isinstance(storage_pid[1], tuple)
        and len(storage_pid[1]) == 5
        and storage_pid[1][0] == "storage"
    ):
        return storage_pid[1]

    raise RuntimeError(f"Unexpected storage descriptor: {storage_pid!r}")


def convert(source_path: str, output_directory: str):
    source_path = os.path.abspath(source_path)
    os.makedirs(output_directory, exist_ok=True)

    with zipfile.ZipFile(source_path, "r") as archive:
        state_dict = _load_state_dict(archive)
        manifest_tensors = []
        data_path = os.path.join(output_directory, "data.bin")

        with open(data_path, "wb") as data_file:
            for name, tensor_spec in state_dict.items():
                storage_tag, storage_class, storage_key, _location, _storage_size = _unpack_storage_pid(tensor_spec.storage)
                if storage_tag != "storage":
                    raise RuntimeError(f"Unexpected storage tag for {name}: {storage_tag}")

                _module_name, storage_type_name = storage_class
                dtype_name, item_size = _dtype_info(storage_type_name)

                if not _is_contiguous(tensor_spec.size, tensor_spec.stride):
                    raise RuntimeError(f"Non-contiguous tensor is not supported: {name}")

                element_count = int(prod(tensor_spec.size))
                byte_count = element_count * item_size
                byte_offset = int(tensor_spec.storage_offset) * item_size

                storage_bytes = archive.read(f"archive/data/{storage_key}")
                tensor_bytes = storage_bytes[byte_offset:byte_offset + byte_count]
                if len(tensor_bytes) != byte_count:
                    raise RuntimeError(
                        f"Unexpected byte count for {name}: expected {byte_count}, got {len(tensor_bytes)}")

                data_offset = data_file.tell()
                data_file.write(tensor_bytes)

                manifest_tensors.append(
                    {
                        "name": name,
                        "dtype": dtype_name,
                        "shape": tensor_spec.size,
                        "elementCount": element_count,
                        "dataOffset": data_offset,
                        "byteCount": byte_count,
                    }
                )

    stat = os.stat(source_path)
    source_last_write_time_utc_ticks = DOTNET_TICKS_AT_UNIX_EPOCH + int(stat.st_mtime_ns // 100)
    manifest = {
        "formatVersion": FORMAT_VERSION,
        "sourcePath": source_path,
        "sourceLength": stat.st_size,
        "sourceLastWriteTimeUtcTicks": source_last_write_time_utc_ticks,
        "tensors": manifest_tensors,
    }

    manifest_path = os.path.join(output_directory, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, indent=2)

    print(f"Converted PyTorch state_dict: {source_path} -> {output_directory}")


def main():
    if len(sys.argv) != 3:
        print("Usage: convert_pytorch_state_dict.py <source.pth> <output-dir>", file=sys.stderr)
        return 2

    convert(sys.argv[1], sys.argv[2])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
