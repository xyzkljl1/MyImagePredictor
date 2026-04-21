using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace ImagePopularity.Core;

public sealed class PopularityModel : Module<Tensor, Tensor>
{
    private readonly Module<Tensor, Tensor> _backbone;
    private readonly Module<Tensor, Tensor> _head;
    private readonly string _backboneName;

    public string BackboneName => _backboneName;

    public PopularityModel(
        PopularityModelConfig? config = null,
        string? backboneWeightsFile = null,
        Device? device = null) : base(nameof(PopularityModel))
    {
        config ??= new PopularityModelConfig();
        _backboneName = PopularityModelConfig.NormalizeBackbone(config.Backbone);

        if (!PopularityModelConfig.IsSupportedBackbone(_backboneName))
        {
            throw new ArgumentOutOfRangeException(nameof(config.Backbone), config.Backbone,
                $"Backbone must be one of: {PopularityBackboneCatalog.SupportedList}.");
        }

        var initialDevice = device ?? CPU;
        _backbone = PopularityBackboneCatalog.CreateBackbone(_backboneName, backboneWeightsFile, initialDevice);
        var features = InferBackboneFeatureDimension(_backbone, initialDevice);

        _head = Sequential(
            ("flatten", Flatten(1)),
            ("fc1", Linear(features, 512)),
            ("relu", ReLU(inplace: true)),
            ("dropout", Dropout(0.35)),
            ("fc2", Linear(512, 1)));

        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        var x = _backbone.forward(input);
        return _head.forward(x);
    }

    public void SetBackboneTrainable(bool trainable)
    {
        foreach (var parameter in _backbone.parameters())
        {
            parameter.requires_grad_(trainable);
        }

        if (trainable)
        {
            _backbone.train();
        }
        else
        {
            _backbone.eval();
        }
    }

    private static int InferBackboneFeatureDimension(Module<Tensor, Tensor> backbone, Device device)
    {
        using var noGrad = torch.no_grad();
        using var sample = torch.zeros(new long[] { 1, 3, 224, 224 }, device: device);
        using var output = backbone.forward(sample);

        if (output.shape.Length < 2)
        {
            throw new InvalidOperationException(
                $"Backbone output rank is too small to infer feature dimension. Shape: [{string.Join(", ", output.shape)}]");
        }

        long featureCount = 1;
        for (var i = 1; i < output.shape.Length; i++)
        {
            featureCount = checked(featureCount * output.shape[i]);
        }

        if (featureCount <= 0 || featureCount > int.MaxValue)
        {
            throw new InvalidOperationException(
                $"Backbone feature dimension is invalid: {featureCount}. Shape: [{string.Join(", ", output.shape)}]");
        }

        return (int)featureCount;
    }
}
