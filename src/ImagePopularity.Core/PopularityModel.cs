using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using System.Reflection;

namespace ImagePopularity.Core;

public sealed class PopularityModel : Module<Tensor, Tensor>
{
    private readonly Module<Tensor, Tensor> _backbone;
    private readonly Module<Tensor, Tensor> _head;
    private readonly string _backboneName;
    private readonly IReadOnlyList<BackboneStage> _backboneStages;

    public string BackboneName => _backboneName;

    public int BackboneStageCount => _backboneStages.Count;

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
        _backboneStages = BuildBackboneStages(_backbone, _backboneName);
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

    public void ConfigureBackboneTrainability(int trainableStageCount)
    {
        if (_backboneStages.Count == 0)
        {
            SetBackboneTrainable(trainableStageCount > 0);
            return;
        }

        var clampedStageCount = Math.Clamp(trainableStageCount, 0, _backboneStages.Count);
        var firstTrainableIndex = _backboneStages.Count - clampedStageCount;

        for (var i = 0; i < _backboneStages.Count; i++)
        {
            var trainable = i >= firstTrainableIndex;
            var stage = _backboneStages[i];

            foreach (var parameter in GetStageParameters(stage))
            {
                parameter.requires_grad_(trainable);
            }

            foreach (var module in stage.Modules)
            {
                if (trainable)
                {
                    ((dynamic)module).train();
                }
                else
                {
                    ((dynamic)module).eval();
                }
            }
        }
    }

    public IReadOnlyList<string> GetTrainableBackboneStageNames(int trainableStageCount)
    {
        if (_backboneStages.Count == 0)
        {
            return trainableStageCount > 0
                ? ["backbone"]
                : Array.Empty<string>();
        }

        var clampedStageCount = Math.Clamp(trainableStageCount, 0, _backboneStages.Count);
        return _backboneStages
            .Skip(_backboneStages.Count - clampedStageCount)
            .Select(stage => stage.Name)
            .ToArray();
    }

    internal IReadOnlyList<LayerwiseParameterGroup> GetLayerwiseParameterGroups(int trainableBackboneStageCount)
    {
        var groups = new List<LayerwiseParameterGroup>();

        var headParameters = _head.parameters().ToArray();
        if (headParameters.Length > 0)
        {
            groups.Add(new LayerwiseParameterGroup("head", headParameters, 1.0));
        }

        if (_backboneStages.Count == 0)
        {
            var backboneParameters = _backbone.parameters()
                .Where(parameter => parameter.requires_grad)
                .ToArray();
            if (backboneParameters.Length > 0)
            {
                groups.Add(new LayerwiseParameterGroup("backbone", backboneParameters, 0.5));
            }

            return groups;
        }

        var clampedStageCount = Math.Clamp(trainableBackboneStageCount, 0, _backboneStages.Count);
        foreach (var stage in _backboneStages.Skip(_backboneStages.Count - clampedStageCount))
        {
            var trainableParameters = GetStageParameters(stage)
                .Where(parameter => parameter.requires_grad)
                .ToArray();
            if (trainableParameters.Length == 0)
            {
                continue;
            }

            groups.Add(new LayerwiseParameterGroup(stage.Name, trainableParameters, GetBackboneStageLearningRateScale(stage.Name)));
        }

        return groups;
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

    private static IReadOnlyList<BackboneStage> BuildBackboneStages(Module<Tensor, Tensor> backbone, string backboneName)
    {
        if (!backboneName.StartsWith("resnet", StringComparison.Ordinal))
        {
            return Array.Empty<BackboneStage>();
        }

        var stages = new List<BackboneStage>();

        var stemModules = new List<object>();
        AddIfModuleExists(stemModules, backbone, "conv1");
        AddIfModuleExists(stemModules, backbone, "bn1");
        AddStageIfAny(stages, "stem", stemModules);

        foreach (var stageName in new[] { "layer1", "layer2", "layer3", "layer4" })
        {
            var modules = new List<object>();
            AddIfModuleExists(modules, backbone, stageName);
            AddStageIfAny(stages, stageName, modules);
        }

        return stages;
    }

    private static void AddStageIfAny(
        IList<BackboneStage> stages,
        string stageName,
        IReadOnlyList<object> modules)
    {
        if (modules.Count == 0)
        {
            return;
        }

        var hasAnyParameters = false;
        foreach (var module in modules)
        {
            foreach (var _ in ((dynamic)module).parameters())
            {
                hasAnyParameters = true;
                break;
            }

            if (hasAnyParameters)
            {
                break;
            }
        }

        if (!hasAnyParameters)
        {
            return;
        }

        stages.Add(new BackboneStage(stageName, modules.ToArray()));
    }

    private static void AddIfModuleExists(ICollection<object> destination, object instance, string memberName)
    {
        var value = TryGetMemberValue(instance, memberName);
        if (value is not null)
        {
            destination.Add(value);
        }
    }

    private static object? TryGetMemberValue(object instance, string memberName)
    {
        const BindingFlags flags = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.IgnoreCase;

        var property = instance.GetType().GetProperty(memberName, flags);
        if (property?.GetValue(instance) is { } propertyValue)
        {
            return propertyValue;
        }

        var field = instance.GetType().GetField(memberName, flags);
        return field?.GetValue(instance);
    }

    private static double GetBackboneStageLearningRateScale(string stageName)
    {
        return stageName switch
        {
            "layer4" => 0.75,
            "layer3" => 0.5,
            "layer2" => 0.25,
            "layer1" => 0.15,
            "stem" => 0.10,
            _ => 0.25
        };
    }

    private static IEnumerable<TorchSharp.Modules.Parameter> GetStageParameters(BackboneStage stage)
    {
        var seen = new HashSet<TorchSharp.Modules.Parameter>(ReferenceEqualityComparer.Instance);
        foreach (var module in stage.Modules)
        {
            foreach (var parameter in ((dynamic)module).parameters())
            {
                if (parameter is TorchSharp.Modules.Parameter typedParameter && seen.Add(typedParameter))
                {
                    yield return typedParameter;
                }
            }
        }
    }

    internal sealed record LayerwiseParameterGroup(string Name, TorchSharp.Modules.Parameter[] Parameters, double LearningRateScale);

    private sealed record BackboneStage(string Name, object[] Modules);
}
