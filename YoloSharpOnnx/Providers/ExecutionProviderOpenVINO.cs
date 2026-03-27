using Microsoft.ML.OnnxRuntime;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YoloSharpOnnx.Inference;
using YoloSharpOnnx.Models;

namespace YoloSharpOnnx.Providers
{
    public class ExecutionProviderOpenVINO : ExecutionProvider, IExecutionProvider
    {
        private IntelDeviceType _intelDeviceType;

        public ExecutionProviderOpenVINO(string modelPath, IntelDeviceType intelDeviceType) : base(modelPath)
        {
            _intelDeviceType = intelDeviceType;
        }

        public IYoloDetect CreateYoloDetect()
        {
            throw new NotImplementedException();
        }

        public override IYoloDetect GetYoloDetector(InferenceSession session, SessionOptions options, IPostprocess postprocess, OnnxModel onnxModel)
        {
            throw new NotImplementedException();
        }
    }
}
