using Microsoft.ML.OnnxRuntime;
using System;
using System.Collections.Generic;
using System.Text;
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
    public class ExecutionProviderTensorRT : ExecutionProvider
    {
        private int _deviceId;
        private Dictionary<string, string> _providerOptionsDict;

        public ExecutionProviderTensorRT(string modelPath, int deviceId)
            : this(modelPath, deviceId, null)
        {

        }
        public ExecutionProviderTensorRT(string modelPath, int deviceId, Dictionary<string, string> providerOptionsDict = null)
            : this(modelPath, deviceId, providerOptionsDict, null)
        {

        }
        public ExecutionProviderTensorRT(string modelPath, int deviceId, Dictionary<string, string> providerOptionsDict = null, SessionOptions sessionOptions = null)
            : base(modelPath, sessionOptions)
        {
            _deviceId = deviceId;
            _providerOptionsDict = providerOptionsDict;
            BuildInferenceSession();
        }

        internal override DeviceType GetDeviceType()
        {
            return DeviceType.GPU;
        }

        internal override IYoloDetectCore<DetectionResult, DetectionBatchResult> GetYoloDetector(InferenceSession session, IDetCorePostprocess<DetectionResult> postprocess, IDetPreprocess preprocess, OnnxModel onnxModel)
        {
            return new YoloDetectIoBinding(session, postprocess, preprocess, onnxModel, YoloConfiguration);
        }

        internal override IYoloClassify GetYoloClassify(InferenceSession session, IClsPostprocess postprocess, IClsPreprocess preprocess, OnnxModel onnxModel)
        {
            return new YoloClsIoBinding(session, onnxModel, YoloConfiguration, postprocess, preprocess);
        }

        internal override InferenceSession BuildSessionOptions(SessionOptions sessionOptions)
        {
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
                using var tensorrtProviderOptions = new OrtTensorRTProviderOptions();
                tensorrtProviderOptions.UpdateOptions(_providerOptionsDict);
                sessionOptions.AppendExecutionProvider_Tensorrt(tensorrtProviderOptions);
            }
            else
            {
                sessionOptions.AppendExecutionProvider_Tensorrt(_deviceId);
            }

            return new InferenceSession(ModelPath, sessionOptions);
        }

        internal override IYoloSegment GetYoloSegment(InferenceSession session, ISegPostprocess postprocess, IDetPreprocess preprocess, OnnxModel onnxModel)
        {
            return new YoloSegIoBinding(session, postprocess, preprocess, onnxModel, YoloConfiguration);
        }

        internal override IYoloDetectCore<PoseResult, PoseBatchResult> GetYoloPose(InferenceSession session, IDetCorePostprocess<PoseResult> postprocess, IDetPreprocess preprocess, OnnxModel onnxModel)
        {
            return new YoloPoseIoBinding(session, postprocess, preprocess, onnxModel, YoloConfiguration);
        }

        internal override IYoloDetectCore<ObbResult, ObbBatchResult> GetYoloObb(InferenceSession session, IDetCorePostprocess<ObbResult> postprocess, IDetPreprocess preprocess, OnnxModel onnxModel)
        {
            return new YoloObbIoBinding(session, postprocess, preprocess, onnxModel, YoloConfiguration);
        }
    }
}
