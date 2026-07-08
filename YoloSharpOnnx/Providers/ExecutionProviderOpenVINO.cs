using Microsoft.ML.OnnxRuntime;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Inference.Classify;
using YoloSharpOnnx.Inference.Detect;
using YoloSharpOnnx.Inference.DetectCore;
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

        public ExecutionProviderOpenVINO(string modelPath, IntelDeviceType intelDeviceType)
            : this(modelPath, intelDeviceType, null)
        {
        }
        public ExecutionProviderOpenVINO(string modelPath, IntelDeviceType intelDeviceType, SessionOptions sessionOptions)
            : base(modelPath, sessionOptions)
        {
            _intelDeviceType = intelDeviceType;
            BuildInferenceSession();
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

        internal override IYoloDetectCore<DetectionResult, DetectionBatchResult> GetYoloDetector(InferenceSession session, IDetCorePostprocess<DetectionResult> postprocess, IDetPreprocess preprocess, OnnxModel onnxModel)
        {
            if (_intelDeviceType == IntelDeviceType.CPU)
            {
                return new YoloDetectOrtVal(session, postprocess, preprocess, onnxModel, YoloConfiguration);
            }
            else
            {
                return new YoloDetectIoBinding(session, postprocess, preprocess, onnxModel, YoloConfiguration);
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

        internal override IYoloClassify GetYoloClassify(InferenceSession session, IClsPostprocess postprocess, IClsPreprocess preprocess, OnnxModel onnxModel)
        {
            if (_intelDeviceType == IntelDeviceType.CPU)
            {
                return new YoloClsOrtVal(session, onnxModel, YoloConfiguration, postprocess, preprocess);
            }
            else
            {
                return new YoloClsIoBinding(session, onnxModel, YoloConfiguration, postprocess, preprocess);
            }
        }

        internal override InferenceSession BuildSessionOptions(SessionOptions sessionOptions)
        {
            sessionOptions.AppendExecutionProvider_OpenVINO(GetIntelDeviceType());
            return new InferenceSession(ModelPath, sessionOptions);
        }

        internal override IYoloSegment GetYoloSegment(InferenceSession session, ISegPostprocess postprocess, IDetPreprocess preprocess, OnnxModel onnxModel)
        {
            if (_intelDeviceType == IntelDeviceType.CPU)
            {
                return new YoloSegOrtVal(session, postprocess, preprocess, onnxModel, YoloConfiguration);
            }
            else
            {
                return new YoloSegIoBinding(session, postprocess, preprocess, onnxModel, YoloConfiguration);
            }
        }

        internal override IYoloDetectCore<PoseResult, PoseBatchResult> GetYoloPose(InferenceSession session, IDetCorePostprocess<PoseResult> postprocess, IDetPreprocess preprocess, OnnxModel onnxModel)
        {
            if (_intelDeviceType == IntelDeviceType.CPU)
            {
                return new YoloPoseOrtVal(session, postprocess, preprocess, onnxModel, YoloConfiguration);
            }
            else
            {
                return new YoloPoseIoBinding(session, postprocess, preprocess, onnxModel, YoloConfiguration);
            }
        }

        internal override IYoloDetectCore<ObbResult, ObbBatchResult> GetYoloObb(InferenceSession session, IDetCorePostprocess<ObbResult> postprocess, IDetPreprocess preprocess, OnnxModel onnxModel)
        {
            if (_intelDeviceType == IntelDeviceType.CPU)
            {
                return new YoloObbOrtVal(session, postprocess, preprocess, onnxModel, YoloConfiguration);
            }
            else
            {
                return new YoloObbIoBinding(session, postprocess, preprocess, onnxModel, YoloConfiguration);
            }
        }
    }
}
