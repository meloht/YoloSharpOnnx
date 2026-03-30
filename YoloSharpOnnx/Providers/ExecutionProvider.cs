using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Reflection.Emit;
using System.Text;
using YoloSharpOnnx.Inference;
using YoloSharpOnnx.Models;

namespace YoloSharpOnnx.Providers
{
    public abstract class ExecutionProvider
    {
        private const string End2End = "end2end";
        private const string OnnxNames = "names";

        public string ModelPath { get; set; }

        protected abstract IYoloDetect GetYoloDetector(InferenceSession session, SessionOptions options, IPostprocess postprocess, IPreprocess preprocess, OnnxModel onnxModel);
        protected abstract DeviceType GetDeviceType();

        public ExecutionProvider(string modelPath)
        {
            ModelPath = modelPath;
        }

        protected IYoloDetect BuildInferenceSession(SessionOptions options)
        {
            InferenceSession session = new InferenceSession(ModelPath, options);

            OnnxModel onnxModel = ParseOnnxModel(session);

            var postprocess = GetPostprocessor(onnxModel);
            var preprocess = GetPreprocess(onnxModel);

            return GetYoloDetector(session, options, postprocess, preprocess, onnxModel);
        }

        private IPostprocess GetPostprocessor(OnnxModel onnxModel)
        {
            if (onnxModel.IsEndToEnd)
            {
                return new PostprocessEndToEnd(onnxModel.Labels);
            }
            return new PostprocessNMS(onnxModel.BoxNum, onnxModel.Labels);
        }

        protected IPreprocess GetPreprocess(OnnxModel onnxModel)
        {
            return new PreprocessComm(onnxModel);
        }

        protected OnnxModel ParseOnnxModel(InferenceSession session)
        {
            OnnxModel model = new OnnxModel();

            model.InputName = session.InputNames[0];
            model.OutputName = session.OutputNames[0];
            model.DeviceType = GetDeviceType();
            var inputMeta = session.InputMetadata;
            var outputMeta = session.OutputMetadata;

            model.InputShape = Array.ConvertAll<int, long>(inputMeta[model.InputName].Dimensions, Convert.ToInt64);
            model.OutputShape = Array.ConvertAll<int, long>(outputMeta[model.OutputName].Dimensions, Convert.ToInt64);

            model.InputHeight = (int)model.InputShape[2];
            model.InputWidth = (int)model.InputShape[3];

            model.InputShapeSize = ShapeUtils.GetSizeForShape(model.InputShape);
            model.OutputShapeSize = ShapeUtils.GetSizeForShape(model.OutputShape);

            model.InputSizeInBytes = model.InputShapeSize * sizeof(float);

            model.Labels = GetModelLabels(session);
            model.ColorPalette = GenerateColorPalette(model.Labels.Length);

            var metaData = session.ModelMetadata.CustomMetadataMap;

            bool isEndToEnd = false;
            if (metaData.ContainsKey(End2End))
            {
                isEndToEnd = bool.Parse(metaData[End2End]);
            }
            model.IsEndToEnd = isEndToEnd;

            model.BoxNum = outputMeta[model.OutputName].Dimensions[2];

            return model;
        }

        protected LabelModel[] GetModelLabels(InferenceSession session)
        {
            var metaData = session.ModelMetadata.CustomMetadataMap;
            var onnxLabelData = metaData[OnnxNames];
            // Labels to Dictionary
            var onnxLabels = onnxLabelData
                .Trim('{', '}')
                .Replace("'", "")
                .Split(", ")
                .Select(x => x.Split(": "))
                .ToDictionary(x => int.Parse(x[0]), x => x[1]);

            return [.. onnxLabels!.Select((label, index) => new LabelModel
            {
                Index = index,
                Name = label.Value,
            })];
        }

        protected Scalar[] GenerateColorPalette(int count)
        {
            var rng = new Random();
            var palette = new Scalar[count];
            var colors = ColorTemplate.Get();
            for (int i = 0; i < count; i++)
            {
                palette[i] = ColorTemplate.HexToRgbaScalar(colors[i % count]);
            }
            return palette;
        }
    }
}
