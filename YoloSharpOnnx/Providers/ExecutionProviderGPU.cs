using Microsoft.ML.OnnxRuntime;
using System;
using System.Collections.Generic;
using System.Text;
using YoloSharpOnnx.Inference;
using YoloSharpOnnx.Models;
using static System.Collections.Specialized.BitVector32;

namespace YoloSharpOnnx.Providers
{
    public class ExecutionProviderGPU : ExecutionProvider, IExecutionProvider
    {
        private int _deviceId;
        private Dictionary<string, string> _providerOptionsDict;


        public ExecutionProviderGPU(string modelPath) : this(modelPath, 0)
        {

        }
        public ExecutionProviderGPU(string modelPath, int deviceId) : this(modelPath, deviceId, [])
        {

        }
        public ExecutionProviderGPU(string modelPath, int deviceId, Dictionary<string, string> providerOptionsDict) : base(modelPath)
        {
            this._deviceId = deviceId;
            this._providerOptionsDict = providerOptionsDict;

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

            return BuildInferenceSession(options);
        }

        public override IYoloDetect GetYoloDetector(InferenceSession session, SessionOptions options, IPostprocess postprocess, OnnxModel onnxModel)
        {
            return new YoloDetectIoBinding(session, options, postprocess, onnxModel);
        }
    }
}
