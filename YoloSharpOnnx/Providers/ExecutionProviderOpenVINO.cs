using Microsoft.ML.OnnxRuntime;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YoloSharpOnnx.Inference.Classify;
using YoloSharpOnnx.Inference.Detect;
using YoloSharpOnnx.Inference.Obb;
using YoloSharpOnnx.Inference.Pose;
using YoloSharpOnnx.Inference.Segment;
using YoloSharpOnnx.Models;

namespace YoloSharpOnnx.Providers
{
    public class ExecutionProviderOpenVINO : ExecutionProvider
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

        internal override DeviceType GetDeviceType()
        {
            if (_intelDeviceType == IntelDeviceType.CPU)
            {
                return DeviceType.CPU;
            }
            else if (_intelDeviceType == IntelDeviceType.NPU)
            {
                return DeviceType.NPU;
            }
            return DeviceType.GPU;
        }

        internal override IYoloDetect GetYoloDetector(InferenceSession session, SessionOptions options, IDetPostprocess postprocess, IDetPreprocess preprocess, OnnxModel onnxModel)
        {
            if (_intelDeviceType == IntelDeviceType.CPU)
            {
                return new YoloDetectOrtVal(session, options, postprocess, preprocess, onnxModel, YoloConfiguration);
            }
            else
            {
                return new YoloDetectIoBinding(session, options, postprocess, preprocess, onnxModel, YoloConfiguration);
            }
        }

        private string GetIntelDeviceType()
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

        internal override IYoloClassify GetYoloClassify(InferenceSession session, SessionOptions options, IClsPostprocess postprocess, IClsPreprocess preprocess, OnnxModel onnxModel)
        {
            if (_intelDeviceType == IntelDeviceType.CPU)
            {
                return new YoloClsOrtVal(session, options, onnxModel, YoloConfiguration, postprocess, preprocess);
            }
            else
            {
                return new YoloClsIoBinding(session, options, onnxModel, YoloConfiguration, postprocess, preprocess);
            }
        }

        internal override SessionOptions BuildSessionOptions()
        {
            SessionOptions options = new SessionOptions();
            options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
            options.EnableCpuMemArena = true;
            options.AppendExecutionProvider_OpenVINO(GetIntelDeviceType());
            return options;
        }

        internal override IYoloSegment GetYoloSegment(InferenceSession session, SessionOptions options, ISegPostprocess postprocess, IDetPreprocess preprocess, OnnxModel onnxModel)
        {
            if (_intelDeviceType == IntelDeviceType.CPU)
            {
                return new YoloSegOrtVal(session, options, postprocess, preprocess, onnxModel, YoloConfiguration);
            }
            else
            {
                return new YoloSegIoBinding(session, options, postprocess, preprocess, onnxModel, YoloConfiguration);
            }
        }

        internal override IYoloPose GetYoloPose(InferenceSession session, SessionOptions options, IPosePostprocess postprocess, IDetPreprocess preprocess, OnnxModel onnxModel)
        {
            if (_intelDeviceType == IntelDeviceType.CPU)
            {
                return new YoloPoseOrtVal(session, options, postprocess, preprocess, onnxModel, YoloConfiguration);
            }
            else
            {
                return new YoloPoseIoBinding(session, options, postprocess, preprocess, onnxModel, YoloConfiguration);
            }
        }

        internal override IYoloObb GetYoloObb(InferenceSession session, SessionOptions options, IObbPostprocess postprocess, IDetPreprocess preprocess, OnnxModel onnxModel)
        {
            if (_intelDeviceType == IntelDeviceType.CPU)
            {
                return new YoloObbOrtVal(session, options, postprocess, preprocess, onnxModel, YoloConfiguration);
            }
            else
            {
                return new YoloObbIoBinding(session, options, postprocess, preprocess, onnxModel, YoloConfiguration);
            }
        }
    }
}
