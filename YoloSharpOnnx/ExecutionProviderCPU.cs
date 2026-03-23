using Microsoft.ML.OnnxRuntime;
using System;
using System.Collections.Generic;
using System.Text;

namespace YoloSharpOnnx
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
    }
}
