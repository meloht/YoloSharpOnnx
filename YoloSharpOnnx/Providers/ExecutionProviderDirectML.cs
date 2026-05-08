using Microsoft.ML.OnnxRuntime;
using System;
using System.Collections.Generic;
using System.Text;
using YoloSharpOnnx.Inference.Detect;
using YoloSharpOnnx.Models;

namespace YoloSharpOnnx.Providers
{
    public class ExecutionProviderDirectML : ExecutionProvider, IExecutionProvider
    {
        private int _deviceId;
        public ExecutionProviderDirectML(string modelPath) : this(modelPath, 0)
        {
        }

        public ExecutionProviderDirectML(string modelPath, int deviceId) : base(modelPath)
        {
            this._deviceId = deviceId;


        }

        public IYoloDetect CreateYoloDetect()
        {
            SessionOptions sessionOptions = new SessionOptions();
            sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
            sessionOptions.AppendExecutionProvider_DML(this._deviceId);
            sessionOptions.EnableCpuMemArena = true;

            return BuildInferenceSession(sessionOptions);
        }

        protected override DeviceType GetDeviceType()
        {
            return DeviceType.GPU;
        }

        protected override IYoloDetect GetYoloDetector(InferenceSession session, SessionOptions options, IDetPostprocess postprocess, IDetPreprocess preprocess, OnnxModel onnxModel)
        {
            return new YoloDetectIoBinding(session, options, postprocess, preprocess, onnxModel, YoloConfiguration);
        }

        public void SetYoloConfiguration(YoloConfig yoloConfig)
        {
            SetYoloConfig(yoloConfig);
        }
    }
}
