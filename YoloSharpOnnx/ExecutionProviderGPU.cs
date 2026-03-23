using Microsoft.ML.OnnxRuntime;
using System;
using System.Collections.Generic;
using System.Text;
using static System.Collections.Specialized.BitVector32;

namespace YoloSharpOnnx
{
    public class ExecutionProviderGPU : ExecutionProvider, IExecutionProvider
    {
        public int DeviceId { get; set; }
        public Dictionary<string, string> ProviderOptionsDict { get; set; } = [];


        public ExecutionProviderGPU(string modelPath) : this(modelPath, 0)
        {

        }
        public ExecutionProviderGPU(string modelPath, int deviceId) : this(modelPath, deviceId, [])
        {

        }
        public ExecutionProviderGPU(string modelPath, int deviceId, Dictionary<string, string> providerOptionsDict) : base(modelPath)
        {
            this.DeviceId = deviceId;
            this.ProviderOptionsDict = providerOptionsDict;

        }

        public IYoloDetect CreateYoloDetect()
        {
            SessionOptions options;
            if (this.ProviderOptionsDict != null && this.ProviderOptionsDict.Count > 0)
            {
                if (ProviderOptionsDict.ContainsKey("device_id"))
                {
                    ProviderOptionsDict["device_id"] = DeviceId.ToString();
                }
                else
                {
                    ProviderOptionsDict.Add("device_id", DeviceId.ToString());
                }
                var cudaProviderOptions = new OrtCUDAProviderOptions();
                cudaProviderOptions.UpdateOptions(ProviderOptionsDict);
                options = SessionOptions.MakeSessionOptionWithCudaProvider(cudaProviderOptions);
            }
            else
            {
                options = SessionOptions.MakeSessionOptionWithCudaProvider(DeviceId);
            }

            options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
            options.EnableCpuMemArena = true;

            return BuildInferenceSession(options);
        }
    }
}
