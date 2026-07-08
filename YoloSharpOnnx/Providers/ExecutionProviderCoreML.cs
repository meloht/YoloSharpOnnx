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
    public class ExecutionProviderCoreML : ExecutionProvider
    {
        private CoreMLFlags _coreMLFlags;
        public ExecutionProviderCoreML(string modelPath, CoreMLFlags coreMLFlags = CoreMLFlags.COREML_FLAG_USE_NONE)
            : this(modelPath, coreMLFlags, null)
        {
        }
        public ExecutionProviderCoreML(string modelPath, CoreMLFlags coreMLFlags = CoreMLFlags.COREML_FLAG_USE_NONE, SessionOptions sessionOptions = null)
            : base(modelPath, sessionOptions)
        {
            _coreMLFlags = coreMLFlags;
            BuildInferenceSession();
        }

        internal override DeviceType GetDeviceType()
        {
            return DeviceType.CPU;
        }

        internal override IYoloDetectCore<DetectionResult, DetectionBatchResult> GetYoloDetector(InferenceSession session, IDetCorePostprocess<DetectionResult> postprocess, IDetPreprocess preprocess, OnnxModel onnxModel)
        {
            return new YoloDetectOrtVal(session, postprocess, preprocess, onnxModel, YoloConfiguration);
        }

        internal override IYoloClassify GetYoloClassify(InferenceSession session, IClsPostprocess postprocess, IClsPreprocess preprocess, OnnxModel onnxModel)
        {
            return new YoloClsOrtVal(session, onnxModel, YoloConfiguration, postprocess, preprocess);
        }

        internal override InferenceSession BuildSessionOptions(SessionOptions sessionOptions)
        {
            sessionOptions.AppendExecutionProvider_CoreML(_coreMLFlags);
            return new InferenceSession(ModelPath, sessionOptions);
        }

        internal override IYoloSegment GetYoloSegment(InferenceSession session, ISegPostprocess postprocess, IDetPreprocess preprocess, OnnxModel onnxModel)
        {
            return new YoloSegOrtVal(session, postprocess, preprocess, onnxModel, YoloConfiguration);
        }

        internal override IYoloDetectCore<PoseResult, PoseBatchResult> GetYoloPose(InferenceSession session, IDetCorePostprocess<PoseResult> postprocess, IDetPreprocess preprocess, OnnxModel onnxModel)
        {
            return new YoloPoseOrtVal(session, postprocess, preprocess, onnxModel, YoloConfiguration);
        }

        internal override IYoloDetectCore<ObbResult, ObbBatchResult> GetYoloObb(InferenceSession session, IDetCorePostprocess<ObbResult> postprocess, IDetPreprocess preprocess, OnnxModel onnxModel)
        {
            return new YoloObbOrtVal(session, postprocess, preprocess, onnxModel, YoloConfiguration);
        }
    }
}
