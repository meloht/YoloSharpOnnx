using Microsoft.ML.OnnxRuntime;
using System;
using System.Collections.Generic;
using System.Text;
using YoloSharpOnnx.Inference;
using YoloSharpOnnx.Models;

namespace YoloSharpOnnx.Providers
{
    public class ExecutionProviderCoreML : ExecutionProvider, IExecutionProvider
    {
        private CoreMLFlags _coreMLFlags;
        public ExecutionProviderCoreML(string modelPath, CoreMLFlags coreMLFlags = CoreMLFlags.COREML_FLAG_USE_NONE) : base(modelPath)
        {
            _coreMLFlags = coreMLFlags;
        }
        public IYoloDetect CreateYoloDetect()
        {
            SessionOptions sessionOptions = new SessionOptions();
            sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
            sessionOptions.EnableCpuMemArena = true;
            sessionOptions.AppendExecutionProvider_CoreML(_coreMLFlags);
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
