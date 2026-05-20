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
    public class ExecutionProviderCUDA : ExecutionProvider, IExecutionProvider
    {
        private int _deviceId;
        private Dictionary<string, string> _providerOptionsDict;


        public ExecutionProviderCUDA(string modelPath) : this(modelPath, 0)
        {

        }
        public ExecutionProviderCUDA(string modelPath, int deviceId) : this(modelPath, deviceId, null)
        {

        }
        public ExecutionProviderCUDA(string modelPath, int deviceId, Dictionary<string, string> providerOptionsDict = null) : base(modelPath)
        {
            this._deviceId = deviceId;
            this._providerOptionsDict = providerOptionsDict;

        }


        protected override DeviceType GetDeviceType()
        {
            return DeviceType.GPU;
        }

        protected override IYoloDetect GetYoloDetector(InferenceSession session, SessionOptions options, IDetPostprocess postprocess, IDetPreprocess preprocess, OnnxModel onnxModel)
        {
            return new YoloDetectIoBinding(session, options, postprocess, preprocess, onnxModel, YoloConfiguration);
        }


        protected override IYoloClassify GetYoloClassify(InferenceSession session, SessionOptions options, IClsPostprocess postprocess, IClsPreprocess preprocess, OnnxModel onnxModel)
        {
            return new YoloClsIoBinding(session, options, onnxModel, YoloConfiguration, postprocess, preprocess);
        }

        protected override SessionOptions BuildSessionOptions()
        {
            SessionOptions options;
            if (this._providerOptionsDict != null && this._providerOptionsDict.Count > 0)
            {
                if (_providerOptionsDict.ContainsKey("device_id"))
                {
                    _providerOptionsDict["device_id"] = _deviceId.ToString();
                }
                else
                {
                    _providerOptionsDict.Add("device_id", _deviceId.ToString());
                }
                var cudaProviderOptions = new OrtCUDAProviderOptions();
                cudaProviderOptions.UpdateOptions(_providerOptionsDict);
                options = SessionOptions.MakeSessionOptionWithCudaProvider(cudaProviderOptions);
            }
            else
            {
                options = SessionOptions.MakeSessionOptionWithCudaProvider(_deviceId);
            }

            options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
            options.EnableCpuMemArena = true;

            return options;
        }

        protected override IYoloSegment GetYoloSegment(InferenceSession session, SessionOptions options, ISegPostprocess postprocess, IDetPreprocess preprocess, OnnxModel onnxModel)
        {
            return new YoloSegIoBinding(session, options, postprocess, preprocess, onnxModel, YoloConfiguration);
        }

        protected override IYoloPose GetYoloPose(InferenceSession session, SessionOptions options, IPosePostprocess postprocess, IDetPreprocess preprocess, OnnxModel onnxModel)
        {
            return new YoloPoseIoBinding(session, options, postprocess, preprocess, onnxModel, YoloConfiguration);
        }
    }
}
