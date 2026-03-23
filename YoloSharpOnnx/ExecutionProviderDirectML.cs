using Microsoft.ML.OnnxRuntime;
using System;
using System.Collections.Generic;
using System.Text;

namespace YoloSharpOnnx
{
    public class ExecutionProviderDirectML : ExecutionProvider, IExecutionProvider
    {
        public int DeviceId { get; set; } = 0;
        public ExecutionProviderDirectML(string modelPath) : this(modelPath, 0)
        {
        }

        public ExecutionProviderDirectML(string modelPath, int deviceId) : base(modelPath)
        {
            this.DeviceId = deviceId;

           
        }

        public IYoloDetect CreateYoloDetect()
        {
            SessionOptions sessionOptions = new SessionOptions();
            sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
            sessionOptions.AppendExecutionProvider_DML(this.DeviceId);
            sessionOptions.EnableCpuMemArena = true;

            return BuildInferenceSession(sessionOptions);
        }
    }
}
