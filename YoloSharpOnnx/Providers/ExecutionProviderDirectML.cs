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
    public class ExecutionProviderDirectML : ExecutionProvider
    {
        private int _deviceId;
        public ExecutionProviderDirectML(string modelPath) : this(modelPath, 0)
        {
        }

        public ExecutionProviderDirectML(string modelPath, int deviceId) : base(modelPath)
        {
            this._deviceId = deviceId;
        }

        internal override DeviceType GetDeviceType()
        {
            return DeviceType.GPU;
        }

        internal override IYoloDetectCore<DetectionResult, DetectionBatchResult> GetYoloDetector(InferenceSession session, SessionOptions options, IDetCorePostprocess<DetectionResult> postprocess, IDetPreprocess preprocess, OnnxModel onnxModel)
        {
            return new YoloDetectIoBinding(session, options, postprocess, preprocess, onnxModel, YoloConfiguration);
        }


        internal override IYoloClassify GetYoloClassify(InferenceSession session, SessionOptions options, IClsPostprocess postprocess, IClsPreprocess preprocess, OnnxModel onnxModel)
        {
            return new YoloClsIoBinding(session, options,  onnxModel, YoloConfiguration, postprocess, preprocess);
        }

        internal override SessionOptions BuildSessionOptions()
        {
            SessionOptions sessionOptions = new SessionOptions();
            sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
            sessionOptions.AppendExecutionProvider_DML(this._deviceId);
            sessionOptions.EnableCpuMemArena = true;
            sessionOptions.EnableMemoryPattern = true;

            return sessionOptions;
        }

        internal override IYoloSegment GetYoloSegment(InferenceSession session, SessionOptions options, ISegPostprocess postprocess, IDetPreprocess preprocess, OnnxModel onnxModel)
        {
            return new YoloSegIoBinding(session, options, postprocess, preprocess, onnxModel, YoloConfiguration);
        }

        internal override IYoloDetectCore<PoseResult, PoseBatchResult> GetYoloPose(InferenceSession session, SessionOptions options, IDetCorePostprocess<PoseResult> postprocess, IDetPreprocess preprocess, OnnxModel onnxModel)
        {
            return new YoloPoseIoBinding(session, options, postprocess, preprocess, onnxModel, YoloConfiguration);
        }

        internal override IYoloDetectCore<ObbResult, ObbBatchResult> GetYoloObb(InferenceSession session, SessionOptions options, IDetCorePostprocess<ObbResult> postprocess, IDetPreprocess preprocess, OnnxModel onnxModel)
        {
            return new YoloObbIoBinding(session, options, postprocess, preprocess, onnxModel, YoloConfiguration);
        }
    }
}
