using Microsoft.ML.OnnxRuntime;
using System;
using System.Collections.Generic;
using System.Text;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Inference.Classify;
using YoloSharpOnnx.Inference.Detect;
using YoloSharpOnnx.Inference.DetectCore;
using YoloSharpOnnx.Inference.Obb;
using YoloSharpOnnx.Inference.Pose;
using YoloSharpOnnx.Inference.Segment;
using YoloSharpOnnx.Models;

namespace YoloSharpOnnx.Providers
{
    public class ExecutionProviderCPU : ExecutionProvider
    {
        public ExecutionProviderCPU(string modelPath) : base(modelPath)
        {
        }

        internal override DeviceType GetDeviceType()
        {
            return DeviceType.CPU;
        }

        internal override IYoloDetectCore<DetectionResult, DetectionBatchResult> GetYoloDetector(InferenceSession session, SessionOptions options, IDetCorePostprocess<DetectionResult> postprocess, IDetPreprocess preprocess, OnnxModel onnxModel)
        {
            return new YoloDetectOrtVal(session, options, postprocess, preprocess, onnxModel, YoloConfiguration);
        }


        internal override IYoloClassify GetYoloClassify(InferenceSession session, SessionOptions options, IClsPostprocess postprocess, IClsPreprocess preprocess, OnnxModel onnxModel)
        {
            return new YoloClsOrtVal(session, options, onnxModel, YoloConfiguration, postprocess, preprocess);
        }

        internal override SessionOptions BuildSessionOptions()
        {
            SessionOptions sessionOptions = new SessionOptions();
            sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
            sessionOptions.EnableCpuMemArena = true;
            return sessionOptions;
        }

        internal override IYoloSegment GetYoloSegment(InferenceSession session, SessionOptions options, ISegPostprocess postprocess, IDetPreprocess preprocess, OnnxModel onnxModel)
        {
            return new YoloSegOrtVal(session, options, postprocess, preprocess, onnxModel, YoloConfiguration);
        }

        internal override IYoloPose GetYoloPose(InferenceSession session, SessionOptions options, IPosePostprocess postprocess, IDetPreprocess preprocess, OnnxModel onnxModel)
        {
            return new YoloPoseOrtVal(session, options, postprocess, preprocess, onnxModel, YoloConfiguration);
        }

        internal override IYoloObb GetYoloObb(InferenceSession session, SessionOptions options, IObbPostprocess postprocess, IDetPreprocess preprocess, OnnxModel onnxModel)
        {
            return new YoloObbOrtVal(session, options, postprocess, preprocess, onnxModel, YoloConfiguration);
        }
    }
}
