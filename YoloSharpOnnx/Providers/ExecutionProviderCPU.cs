using Microsoft.ML.OnnxRuntime;
using System;
using System.Collections.Generic;
using System.Text;
using YoloSharpOnnx.Inference;
using YoloSharpOnnx.Models;

namespace YoloSharpOnnx.Providers
{
    public class ExecutionProviderCPU : ExecutionProvider, IExecutionProvider
    {
        public ExecutionProviderCPU(string modelPath) : base(modelPath)
        {
        }

        public IYoloDetect CreateYoloDetect()
        {
            SessionOptions sessionOptions = new SessionOptions();
            sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
            sessionOptions.EnableCpuMemArena = true;

            return BuildInferenceSession(sessionOptions);
        }

        protected override DeviceType GetDeviceType()
        {
            return DeviceType.CPU;
        }

        protected override IYoloDetect GetYoloDetector(InferenceSession session, SessionOptions options, IPostprocess postprocess, IPreprocess preprocess, OnnxModel onnxModel)
        {
            return new YoloDetectOrtVal(session, options, postprocess, preprocess, onnxModel);
        }
    }
}
