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
        private const string CPU = "CPU";
        private const string GPU = "GPU";
        private const string GPU0 = "GPU.0";
        private const string GPU1 = "GPU.1";
        private const string NPU = "NPU";
        private IntelDeviceType _intelDeviceType;

        public ExecutionProviderOpenVINO(string modelPath, IntelDeviceType intelDeviceType) : base(modelPath)
        {
            _intelDeviceType = intelDeviceType;
        }

        public IYoloDetect CreateYoloDetect()
        {
            SessionOptions options = new SessionOptions();
            options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
            options.EnableCpuMemArena = true;
            options.AppendExecutionProvider_OpenVINO(GetDeviceType());
            return BuildInferenceSession(options);
        }

        public override IYoloDetect GetYoloDetector(InferenceSession session, SessionOptions options, IPostprocess postprocess, OnnxModel onnxModel)
        {
            if (_intelDeviceType == IntelDeviceType.CPU)
            {
                return new YoloDetectOrtVal(session, options, postprocess, onnxModel);
            }
            else
            {
                return new YoloDetectIoBinding(session, options, postprocess, onnxModel);
            }
        }

        private string GetDeviceType()
        {
            switch (_intelDeviceType)
            {
                case IntelDeviceType.CPU:
                    return CPU;
                case IntelDeviceType.GPU:
                    return GPU;
                case IntelDeviceType.GPU0:
                    return GPU0;
                case IntelDeviceType.GPU1:
                    return GPU1;
                case IntelDeviceType.NPU:
                    return NPU;
                default:
                    return CPU;
            }
        }
    }
}
