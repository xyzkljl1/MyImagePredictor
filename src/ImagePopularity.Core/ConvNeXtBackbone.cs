using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace ImagePopularity.Core;

internal static class ConvNeXtFactory
{
    private static readonly IReadOnlyDictionary<string, ConvNeXtSpec> Specs =
        new Dictionary<string, ConvNeXtSpec>(StringComparer.Ordinal)
        {
            ["convnexttiny"] = new(
                FeatureDimension: 768,
                DefaultWeightsFileName: "convnext_tiny-983f1562.pth",
                Depths: [3, 3, 9, 3],
                Dimensions: [96, 192, 384, 768],
                StochasticDepthProbability: 0.1),
            ["convnextsmall"] = new(
                FeatureDimension: 768,
                DefaultWeightsFileName: "convnext_small-0c510722.pth",
                Depths: [3, 3, 27, 3],
                Dimensions: [96, 192, 384, 768],
                StochasticDepthProbability: 0.4),
            ["convnextbase"] = new(
                FeatureDimension: 1024,
                DefaultWeightsFileName: "convnext_base-6075fbad.pth",
                Depths: [3, 3, 27, 3],
                Dimensions: [128, 256, 512, 1024],
                StochasticDepthProbability: 0.5),
            ["convnextlarge"] = new(
                FeatureDimension: 1536,
                DefaultWeightsFileName: "convnext_large-ea097f82.pth",
                Depths: [3, 3, 27, 3],
                Dimensions: [192, 384, 768, 1536],
                StochasticDepthProbability: 0.5)
        };

    public static Module<Tensor, Tensor> CreateBackbone(string backboneName, string? weightsFile, Device device)
    {
        var spec = Specs[backboneName];
        var backbone = new ConvNeXtFeatureExtractor(spec);
        if (!string.IsNullOrWhiteSpace(weightsFile))
        {
            TryLoadWeights(backbone, weightsFile);
        }

        backbone.to(device);
        return backbone;
    }

    public static int GetFeatureDimension(string backboneName)
    {
        return Specs[backboneName].FeatureDimension;
    }

    public static string GetDefaultWeightsFileName(string backboneName)
    {
        return Specs[backboneName].DefaultWeightsFileName;
    }

    private static void TryLoadWeights(ConvNeXtFeatureExtractor backbone, string weightsFile)
    {
        if (weightsFile.EndsWith(".pth", StringComparison.OrdinalIgnoreCase))
        {
            PyTorchStateDictLoader.TryLoadFloat32StateDict(backbone, weightsFile, "ConvNeXt");
            return;
        }

        try
        {
            backbone.load(weightsFile);
            Console.WriteLine($"Loaded pretrained backbone weights: {weightsFile}");
        }
        catch (ArgumentException ex) when (LooksLikeIncompatiblePyTorchStateDict(weightsFile, ex))
        {
            Console.WriteLine(
                $"Warning: pretrained ConvNeXt weights are stored as an official PyTorch state_dict and cannot be loaded directly by the current TorchSharp ConvNeXt implementation. Continuing with random ConvNeXt initialization. File: {weightsFile}");
        }
    }

    private static bool LooksLikeIncompatiblePyTorchStateDict(string weightsFile, ArgumentException exception)
    {
        return weightsFile.EndsWith(".pth", StringComparison.OrdinalIgnoreCase) &&
               exception.Message.Contains("Mismatched state_dict sizes", StringComparison.Ordinal);
    }

    private sealed record ConvNeXtSpec(
        int FeatureDimension,
        string DefaultWeightsFileName,
        int[] Depths,
        int[] Dimensions,
        double StochasticDepthProbability);

    private sealed class ConvNeXtFeatureExtractor : Module<Tensor, Tensor>
    {
        private readonly Module<Tensor, Tensor> features;
        private readonly Module<Tensor, Tensor> avgpool;
        private readonly Module<Tensor, Tensor> classifier;
        private readonly Module<Tensor, Tensor> classifierNorm;
        private readonly Module<Tensor, Tensor> classifierFlatten;

        internal readonly Module<Tensor, Tensor> stem;
        internal readonly Module<Tensor, Tensor> stage1;
        internal readonly Module<Tensor, Tensor> stage2;
        internal readonly Module<Tensor, Tensor> stage3;
        internal readonly Module<Tensor, Tensor> stage4;

        public ConvNeXtFeatureExtractor(ConvNeXtSpec spec) : base(nameof(ConvNeXtFeatureExtractor))
        {
            var depths = spec.Depths;
            var dims = spec.Dimensions;
            var totalBlocks = depths.Sum();
            var stageBlockId = 0;

            stem = Sequential(
                ("0", Conv2d(3, dims[0], 4, 4, 0, 1, PaddingModes.Zeros, 1, true)),
                ("1", new ConvNeXtLayerNorm2d(dims[0], eps: 1e-6)));

            var stage1Blocks = CreateStage(dims[0], depths[0], ref stageBlockId, totalBlocks, spec.StochasticDepthProbability);
            var downsample1 = CreateDownsample(dims[0], dims[1]);
            var stage2Blocks = CreateStage(dims[1], depths[1], ref stageBlockId, totalBlocks, spec.StochasticDepthProbability);
            var downsample2 = CreateDownsample(dims[1], dims[2]);
            var stage3Blocks = CreateStage(dims[2], depths[2], ref stageBlockId, totalBlocks, spec.StochasticDepthProbability);
            var downsample3 = CreateDownsample(dims[2], dims[3]);
            var stage4Blocks = CreateStage(dims[3], depths[3], ref stageBlockId, totalBlocks, spec.StochasticDepthProbability);

            stage1 = stage1Blocks;
            stage2 = Sequential(("0", downsample1), ("1", stage2Blocks));
            stage3 = Sequential(("0", downsample2), ("1", stage3Blocks));
            stage4 = Sequential(("0", downsample3), ("1", stage4Blocks));

            features = Sequential(
                ("0", stem),
                ("1", stage1Blocks),
                ("2", downsample1),
                ("3", stage2Blocks),
                ("4", downsample2),
                ("5", stage3Blocks),
                ("6", downsample3),
                ("7", stage4Blocks));

            avgpool = AdaptiveAvgPool2d(1);

            classifierNorm = new ConvNeXtLayerNorm2d(dims[^1], eps: 1e-6);
            classifierFlatten = Flatten(1, -1);
            var classifierHead = Linear(dims[^1], 1000, true);
            classifier = Sequential(
                ("0", classifierNorm),
                ("1", classifierFlatten),
                ("2", classifierHead));

            register_module("features", features);
            register_module("avgpool", avgpool);
            register_module("classifier", classifier);
        }

        public override Tensor forward(Tensor input)
        {
            var x = features.forward(input);
            x = avgpool.forward(x);
            x = classifierNorm.forward(x);
            x = classifierFlatten.forward(x);
            return x;
        }

        private static Module<Tensor, Tensor> CreateStage(
            int dimension,
            int depth,
            ref int stageBlockId,
            int totalBlocks,
            double stochasticDepthProbability)
        {
            var modules = new (string, Module<Tensor, Tensor>)[depth];
            for (var i = 0; i < depth; i++)
            {
                var dropProbability = totalBlocks <= 1
                    ? 0d
                    : stochasticDepthProbability * stageBlockId / (totalBlocks - 1d);
                modules[i] = (i.ToString(), new ConvNeXtBlock(dimension, 1e-6, dropProbability));
                stageBlockId++;
            }

            return Sequential(modules);
        }

        private static Module<Tensor, Tensor> CreateDownsample(int inputChannels, int outputChannels)
        {
            return Sequential(
                ("0", new ConvNeXtLayerNorm2d(inputChannels, eps: 1e-6)),
                ("1", Conv2d(inputChannels, outputChannels, 2, 2, 0, 1, PaddingModes.Zeros, 1, true)));
        }
    }

    private sealed class ConvNeXtBlock : Module<Tensor, Tensor>
    {
        private readonly Module<Tensor, Tensor> block;
        private readonly TorchSharp.Modules.Parameter layer_scale;
        private readonly ConvNeXtStochasticDepth stochastic_depth;

        public ConvNeXtBlock(int dimension, double layerScale, double stochasticDepthProbability) : base(nameof(ConvNeXtBlock))
        {
            block = Sequential(
                ("0", Conv2d(dimension, dimension, 7, 1, 3, 1, PaddingModes.Zeros, dimension, true)),
                ("1", new PermuteModule([0, 2, 3, 1])),
                ("2", LayerNorm(dimension, eps: 1e-6)),
                ("3", Linear(dimension, 4 * dimension, true)),
                ("4", GELU()),
                ("5", Linear(4 * dimension, dimension, true)),
                ("6", new PermuteModule([0, 3, 1, 2])));
            layer_scale = Parameter(torch.ones(dimension, 1, 1) * layerScale, requires_grad: true);
            stochastic_depth = new ConvNeXtStochasticDepth(stochasticDepthProbability);

            register_module("block", block);
            register_parameter("layer_scale", layer_scale);
            register_module("stochastic_depth", stochastic_depth);
        }

        public override Tensor forward(Tensor input)
        {
            var result = layer_scale * block.forward(input);
            result = stochastic_depth.forward(result);
            return result + input;
        }
    }

    private sealed class ConvNeXtLayerNorm2d : Module<Tensor, Tensor>
    {
        private readonly long[] normalized_shape;
        private readonly double eps;
        private readonly TorchSharp.Modules.Parameter weight;
        private readonly TorchSharp.Modules.Parameter bias;

        public ConvNeXtLayerNorm2d(int normalizedShape, double eps) : base(nameof(ConvNeXtLayerNorm2d))
        {
            normalized_shape = [normalizedShape];
            this.eps = eps;
            weight = Parameter(torch.ones(normalizedShape), requires_grad: true);
            bias = Parameter(torch.zeros(normalizedShape), requires_grad: true);

            register_parameter("weight", weight);
            register_parameter("bias", bias);
        }

        public override Tensor forward(Tensor input)
        {
            var x = input.permute(0, 2, 3, 1);
            x = functional.layer_norm(x, normalized_shape, weight, bias, eps);
            x = x.permute(0, 3, 1, 2);
            return x;
        }
    }

    private sealed class PermuteModule : Module<Tensor, Tensor>
    {
        private readonly long[] dimensions;

        public PermuteModule(long[] dimensions) : base(nameof(PermuteModule))
        {
            this.dimensions = dimensions;
        }

        public override Tensor forward(Tensor input)
        {
            return input.permute(dimensions);
        }
    }

    private sealed class ConvNeXtStochasticDepth : Module<Tensor, Tensor>
    {
        private readonly double probability;

        public ConvNeXtStochasticDepth(double probability) : base(nameof(ConvNeXtStochasticDepth))
        {
            this.probability = probability;
        }

        public override Tensor forward(Tensor input)
        {
            if (!training || probability <= 0d)
            {
                return input;
            }

            var survivalRate = 1d - probability;
            if (survivalRate <= 0d)
            {
                return torch.zeros_like(input);
            }

            var noiseShape = new long[input.shape.Length];
            noiseShape[0] = input.shape[0];
            for (var i = 1; i < noiseShape.Length; i++)
            {
                noiseShape[i] = 1;
            }

            using var noise = torch.empty(noiseShape, dtype: input.dtype, device: input.device);
            noise.bernoulli_(survivalRate);
            if (survivalRate < 1d)
            {
                noise.div_(survivalRate);
            }

            return input * noise;
        }
    }
}
