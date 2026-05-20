using Microsoft.ML.OnnxRuntime;
using System;
using System.Collections.Generic;
using System.Text;
using YoloSharpOnnx.Inference.Classify;
using YoloSharpOnnx.Inference.Detect;
using YoloSharpOnnx.Inference.Pose;
using YoloSharpOnnx.Inference.Segment;
using YoloSharpOnnx.Models;

namespace YoloSharpOnnx.Providers
{
    public class ExecutionProviderCPU : ExecutionProvider, IExecutionProvider
    {
        public ExecutionProviderCPU(string modelPath) : base(modelPath)
        {
        }

        protected override DeviceType GetDeviceType()
        {
            return DeviceType.CPU;
        }

        protected override IYoloDetect GetYoloDetector(InferenceSession session, SessionOptions options, IDetPostprocess postprocess, IDetPreprocess preprocess, OnnxModel onnxModel)
        {
            return new YoloDetectOrtVal(session, options, postprocess, preprocess, onnxModel, YoloConfiguration);
        }


        protected override IYoloClassify GetYoloClassify(InferenceSession session, SessionOptions options, IClsPostprocess postprocess, IClsPreprocess preprocess, OnnxModel onnxModel)
        {
            return new YoloClsOrtVal(session, options, onnxModel, YoloConfiguration, postprocess, preprocess);
        }

        protected override SessionOptions BuildSessionOptions()
        {
            SessionOptions sessionOptions = new SessionOptions();
            sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
            sessionOptions.EnableCpuMemArena = true;
            return sessionOptions;
        }

        protected override IYoloSegment GetYoloSegment(InferenceSession session, SessionOptions options, ISegPostprocess postprocess, IDetPreprocess preprocess, OnnxModel onnxModel)
        {
            return new YoloSegOrtVal(session, options, postprocess, preprocess, onnxModel, YoloConfiguration);
        }

        protected override IYoloPose GetYoloPose(InferenceSession session, SessionOptions options, IPosePostprocess postprocess, IDetPreprocess preprocess, OnnxModel onnxModel)
        {
            return new YoloPoseOrtVal(session, options, postprocess, preprocess, onnxModel, YoloConfiguration);
        }
    }
}
