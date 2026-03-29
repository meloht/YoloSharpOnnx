using Microsoft.ML.OnnxRuntime;
using System;
using System.Collections.Generic;
using System.Text;
using YoloSharpOnnx.Inference;
using YoloSharpOnnx.Models;

namespace YoloSharpOnnx.Providers
{
    public class ExecutionProviderTensorRT : ExecutionProvider, IExecutionProvider
    {
        private int _deviceId;
        private Dictionary<string, string> _providerOptionsDict;
        public ExecutionProviderTensorRT(string modelPath, int deviceId, Dictionary<string, string> providerOptionsDict) : base(modelPath)
        {
            _deviceId = deviceId;
            _providerOptionsDict = providerOptionsDict;
        }

        public IYoloDetect CreateYoloDetect()
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
                var tensorrtProviderOptions = new OrtTensorRTProviderOptions();
                tensorrtProviderOptions.UpdateOptions(_providerOptionsDict);
                options = SessionOptions.MakeSessionOptionWithTensorrtProvider(tensorrtProviderOptions);
            }
            else
            {
                options = SessionOptions.MakeSessionOptionWithTensorrtProvider(_deviceId);
            }
        
            options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
            options.EnableCpuMemArena = true;

            return BuildInferenceSession(options);
        }

        protected override DeviceType GetDeviceType()
        {
            return DeviceType.GPU;
        }

        protected override IYoloDetect GetYoloDetector(InferenceSession session, SessionOptions options, IPostprocess postprocess, OnnxModel onnxModel)
        {
            return new YoloDetectIoBinding(session, options, postprocess, onnxModel);
        }
    }
}
