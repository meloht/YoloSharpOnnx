using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Reflection.Emit;
using System.Text;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Inference;
using YoloSharpOnnx.Inference.Classify;
using YoloSharpOnnx.Inference.Detect;
using YoloSharpOnnx.Inference.DetectCore;
using YoloSharpOnnx.Inference.Obb;
using YoloSharpOnnx.Inference.Pose;
using YoloSharpOnnx.Inference.Segment;
using YoloSharpOnnx.Models;

namespace YoloSharpOnnx.Providers
{
    public abstract class ExecutionProvider 
    {
        private const string End2End = "end2end";
        private const string OnnxNames = "names";
        private const string ModelTask = "task";
        private const string kpt_shape = "kpt_shape";
        private const string kpt_names = "kpt_names";

        public string ModelPath { get; set; }
        protected YoloConfig YoloConfiguration { get; private set; }
        internal ModelType CurrentModelType { get; private set; }

        internal abstract SessionOptions BuildSessionOptions();

        internal abstract IYoloDetectCore<DetectionResult,DetectionBatchResult> GetYoloDetector(InferenceSession session, SessionOptions options, IDetCorePostprocess<DetectionResult> postprocess, IDetPreprocess preprocess, OnnxModel onnxModel);
        internal abstract IYoloClassify GetYoloClassify(InferenceSession session, SessionOptions options, IClsPostprocess postprocess, IClsPreprocess preprocess, OnnxModel onnxModel);

        internal abstract IYoloSegment GetYoloSegment(InferenceSession session, SessionOptions options, ISegPostprocess postprocess, IDetPreprocess preprocess, OnnxModel onnxModel);
        internal abstract IYoloDetectCore<PoseResult, PoseBatchResult> GetYoloPose(InferenceSession session, SessionOptions options, IDetCorePostprocess<PoseResult> postprocess, IDetPreprocess preprocess, OnnxModel onnxModel);
        internal abstract IYoloDetectCore<ObbResult, ObbBatchResult> GetYoloObb(InferenceSession session, SessionOptions options, IDetCorePostprocess<ObbResult> postprocess, IDetPreprocess preprocess, OnnxModel onnxModel);

        internal abstract DeviceType GetDeviceType();
        private readonly Random _rand;
        public ExecutionProvider(string modelPath)
        {
            ModelPath = modelPath;
            _rand = new Random(0);
        }

        internal void SetYoloConfiguration(YoloConfig yoloConfig)
        {
            YoloConfiguration = yoloConfig;
        }

        internal IYoloDetectCore<DetectionResult, DetectionBatchResult> CreateYoloDetect()
        {
            SessionOptions options = BuildSessionOptions();
            InferenceSession session = new InferenceSession(ModelPath, options);
            OnnxModel onnxModel = ParseOnnxModel(session);
            CurrentModelType = onnxModel.ModelType;
            if (CurrentModelType != ModelType.ObjectDetection)
            {
                session.Dispose();
                options.Dispose();
                return null;
            }
            var postprocess = GetDetPostprocessor(onnxModel);
            var preprocess = GetPreprocess(onnxModel);

            return GetYoloDetector(session, options, postprocess, preprocess, onnxModel);
        }

        internal IYoloClassify CreateYoloClassify()
        {
            SessionOptions options = BuildSessionOptions();
            InferenceSession session = new InferenceSession(ModelPath, options);
            OnnxModel onnxModel = ParseOnnxModel(session);

            CurrentModelType = onnxModel.ModelType;
            if (CurrentModelType != ModelType.Classification)
            {
                session.Dispose();
                options.Dispose();
                return null;
            }

            var postprocess = new ClsPostprocess(onnxModel, YoloConfiguration);
            var preprocess = new ClsPreprocess(onnxModel, YoloConfiguration);

            return GetYoloClassify(session, options, postprocess, preprocess, onnxModel);
        }
        internal IYoloSegment CreateYoloSegment()
        {
            SessionOptions options = BuildSessionOptions();
            InferenceSession session = new InferenceSession(ModelPath, options);
            OnnxModel onnxModel = ParseOnnxModel(session);
            CurrentModelType = onnxModel.ModelType;
            if (CurrentModelType != ModelType.Segmentation)
            {
                session.Dispose();
                options.Dispose();
                return null;
            }
            var postprocess = GetSegPostprocessor(onnxModel);
            var preprocess = GetPreprocess(onnxModel);

            return GetYoloSegment(session, options, postprocess, preprocess, onnxModel);
        }

        internal IYoloDetectCore<PoseResult, PoseBatchResult> CreateYoloPose()
        {
            SessionOptions options = BuildSessionOptions();
            InferenceSession session = new InferenceSession(ModelPath, options);
            OnnxModel onnxModel = ParseOnnxModel(session);
            CurrentModelType = onnxModel.ModelType;
            if (CurrentModelType != ModelType.PoseEstimation)
            {
                session.Dispose();
                options.Dispose();
                return null;
            }
            var postprocess = GetPosePostprocessor(onnxModel);
            var preprocess = GetPreprocess(onnxModel);

            return GetYoloPose(session, options, postprocess, preprocess, onnxModel);
        }

        internal IYoloDetectCore<ObbResult, ObbBatchResult> CreateYoloObb()
        {
            SessionOptions options = BuildSessionOptions();
            InferenceSession session = new InferenceSession(ModelPath, options);
            OnnxModel onnxModel = ParseOnnxModel(session);
            CurrentModelType = onnxModel.ModelType;
            if (CurrentModelType != ModelType.ObbDetection)
            {
                session.Dispose();
                options.Dispose();
                return null;
            }
            var postprocess = GetObbPostprocessor(onnxModel);
            var preprocess = GetPreprocess(onnxModel);

            return GetYoloObb(session, options, postprocess, preprocess, onnxModel);
        }

        private IDetCorePostprocess<DetectionResult> GetDetPostprocessor(OnnxModel onnxModel)
        {
            if (onnxModel.IsEndToEnd)
            {
                return new DetPostprocessEndToEnd(onnxModel, YoloConfiguration);
            }
            return new DetPostprocessNMS(onnxModel, YoloConfiguration);
        }

        private ISegPostprocess GetSegPostprocessor(OnnxModel onnxModel)
        {
            if (onnxModel.IsEndToEnd)
            {
                return new SegPostprocessEndToEnd(onnxModel, YoloConfiguration);
            }
            return new SegPostprocessNMS(onnxModel, YoloConfiguration);
        }

        private IDetCorePostprocess<PoseResult> GetPosePostprocessor(OnnxModel onnxModel)
        {
            if (onnxModel.IsEndToEnd)
            {
                return new PosePostprocessEndToEnd(onnxModel, YoloConfiguration);
            }
            return new PosePostprocessNMS(onnxModel, YoloConfiguration);
        }

        private IDetCorePostprocess<ObbResult> GetObbPostprocessor(OnnxModel onnxModel)
        {
            if (onnxModel.IsEndToEnd)
            {
                return new ObbPostprocessEndToEnd(onnxModel, YoloConfiguration);
            }
            return new ObbPostprocessNMS(onnxModel, YoloConfiguration);
        }

        private IDetPreprocess GetPreprocess(OnnxModel onnxModel)
        {
            return new DetPreprocessComm(onnxModel, YoloConfiguration);
        }

        internal OnnxModel ParseOnnxModel(InferenceSession session)
        {
            OnnxModel model = new OnnxModel();

            model.InputName = session.InputNames[0];
            model.OutputName0 = session.OutputNames[0];
            if (session.OutputNames.Count > 1)
            {
                model.OutputName1 = session.OutputNames[1];
            }
            model.DeviceType = GetDeviceType();
            var inputMeta = session.InputMetadata;
            var outputMeta = session.OutputMetadata;

            model.InputShape = Array.ConvertAll<int, long>(inputMeta[model.InputName].Dimensions, Convert.ToInt64);
            model.OutputShape0 = Array.ConvertAll<int, long>(outputMeta[model.OutputName0].Dimensions, Convert.ToInt64);

            if (session.OutputNames.Count > 1)
            {
                model.OutputShape1 = Array.ConvertAll<int, long>(outputMeta[model.OutputName1].Dimensions, Convert.ToInt64);
            }
            model.InputHeight = (int)model.InputShape[2];
            model.InputWidth = (int)model.InputShape[3];

            model.InputShapeSize = ShapeUtils.GetSizeForShape(model.InputShape);
            model.OutputShapeSize0 = ShapeUtils.GetSizeForShape(model.OutputShape0);

            if (session.OutputNames.Count > 1)
            {
                model.OutputShapeSize1 = ShapeUtils.GetSizeForShape(model.OutputShape1);
            }

            model.InputSizeInBytes = model.InputShapeSize * sizeof(float);
            model.OutputSizeInBytes0 = model.OutputShapeSize0 * sizeof(float);

            if (session.OutputNames.Count > 1)
            {
                model.OutputSizeInBytes1 = model.OutputShapeSize1 * sizeof(float);
            }

            model.Labels = GetModelLabels(session);

            var metaData = session.ModelMetadata.CustomMetadataMap;

            bool isEndToEnd = false;
            if (metaData.ContainsKey(End2End))
            {
                isEndToEnd = bool.Parse(metaData[End2End]);
            }
            if (metaData.ContainsKey(ModelTask))
            {
                model.ModelType = GetModelType(metaData[ModelTask].Trim());
            }
            model.IsEndToEnd = isEndToEnd;

            if (model.ModelType != ModelType.Classification)
            {
                model.ColorPalette = GenerateColorPalette(model.Labels.Length);
            }
            if (metaData.ContainsKey(kpt_shape))
            {
                var kptShape = metaData[kpt_shape].Trim('[', ']').Split(',').Select(int.Parse).ToArray();
                model.KPTShape = kptShape;
            }
            if (metaData.ContainsKey(kpt_names))
            {
                model.KPTNames = GetKptNames(metaData[kpt_names]);
            }

            return model;
        }

        private ModelType GetModelType(string task)
        {
            if (ModelType.ObjectDetection.GetDescription() == task)
            {
                return ModelType.ObjectDetection;
            }
            else if (ModelType.Classification.GetDescription() == task)
            {
                return ModelType.Classification;
            }
            else if (ModelType.ObbDetection.GetDescription() == task)
            {
                return ModelType.ObbDetection;
            }
            else if (ModelType.Segmentation.GetDescription() == task)
            {
                return ModelType.Segmentation;
            }
            else if (ModelType.PoseEstimation.GetDescription() == task)
            {
                return ModelType.PoseEstimation;
            }
            else
            {
                throw new ArgumentOutOfRangeException($"model task: {task} is not support");
            }
        }

        private LabelModel[] GetModelLabels(InferenceSession session)
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

            return [.. onnxLabels!.Select((label, index) => new LabelModel(index, label.Value))];
        }
        /// <summary>
        /// "{0: ['nose', 'left_eye', 'right_eye', 'left_ear']}";
        /// </summary>
        /// <param name="kptNamesData"></param>
        /// <returns></returns>
        private string[][] GetKptNames(string kptNamesData)
        {
            List<string[]> kptNamesList = new List<string[]>();
            string text = kptNamesData.Trim('{', '}');

            var arr = text.Split(':', StringSplitOptions.RemoveEmptyEntries);

            foreach (var item in arr)
            {
                if (!item.Contains("["))
                {
                    continue;
                }
                string str = item.Trim();
                var els = str.Trim('[', ']').Replace("'", "").Split(',', StringSplitOptions.RemoveEmptyEntries);
                if (els.Length > 0)
                {
                    kptNamesList.Add(els);
                }
            }
            return [.. kptNamesList];
        }

        private Scalar[] GenerateColorPalette(int count)
        {
            var palette = new Scalar[count];
            var colors = ColorTemplate.Get();
            for (int i = 0; i < count; i++)
            {
                int idx = i % count;
                if (idx < colors.Length)
                {
                    palette[i] = ColorTemplate.HexToRgbaScalar(colors[idx]);
                }
                else
                {
                    palette[i] = GetRandomColor();
                }
            }
            return palette;
        }

        private Scalar GetRandomColor()
        {
            return new Scalar(
                (byte)_rand.Next(0, 256),
                (byte)_rand.Next(0, 256),
                (byte)_rand.Next(0, 256)
            );
        }

    }
}
