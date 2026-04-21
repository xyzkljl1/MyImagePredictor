using TorchSharp;
using static TorchSharp.torch;

namespace ImagePopularity.Core;

internal static class TorchDeviceSelector
{
    public static Device ResolveCuda()
    {
        if (!cuda.is_available())
        {
            throw new InvalidOperationException("CUDA is required but torch.cuda.is_available() is false. Install CUDA LibTorch runtime and ensure NVIDIA driver is ready.");
        }

        return CUDA;
    }
}
