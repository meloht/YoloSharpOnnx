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
        }

        internal override DeviceType GetDeviceType()
        {
            return DeviceType.CPU;
        }

        internal override IYoloDetectCore<DetectionResult, DetectionBatchResult> GetYoloDetector(InferenceSession session, SessionOptions options, IDetCorePostprocess<DetectionResult> postprocess, IDetPreprocess preprocess, OnnxModel onnxModel)
        {
            return new YoloDetectOrtVal(session, options, postprocess, preprocess, onnxModel, YoloConfiguration);
        }

        internal override IYoloClassify GetYoloClassify(InferenceSession session, SessionOptions options, IClsPostprocess postprocess, IClsPreprocess preprocess, OnnxModel onnxModel)
        {
            return new YoloClsOrtVal(session, options, onnxModel, YoloConfiguration, postprocess, preprocess);
        }

        internal override SessionOptions BuildSessionOptions()
        {
            var sessionOptions = BuildSessionOptionsBase();
            sessionOptions.AppendExecutionProvider_CoreML(_coreMLFlags);
            return sessionOptions;
        }

        internal override IYoloSegment GetYoloSegment(InferenceSession session, SessionOptions options, ISegPostprocess postprocess, IDetPreprocess preprocess, OnnxModel onnxModel)
        {
            return new YoloSegOrtVal(session, options, postprocess, preprocess, onnxModel, YoloConfiguration);
        }

        internal override IYoloDetectCore<PoseResult, PoseBatchResult> GetYoloPose(InferenceSession session, SessionOptions options, IDetCorePostprocess<PoseResult> postprocess, IDetPreprocess preprocess, OnnxModel onnxModel)
        {
            return new YoloPoseOrtVal(session, options, postprocess, preprocess, onnxModel, YoloConfiguration);
        }

        internal override IYoloDetectCore<ObbResult, ObbBatchResult> GetYoloObb(InferenceSession session, SessionOptions options, IDetCorePostprocess<ObbResult> postprocess, IDetPreprocess preprocess, OnnxModel onnxModel)
        {
            return new YoloObbOrtVal(session, options, postprocess, preprocess, onnxModel, YoloConfiguration);
        }
    }
}
