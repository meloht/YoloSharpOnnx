using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Inference.Detect.Models;
using YoloSharpOnnx.Models;

namespace YoloSharpOnnx.Inference.DetectCore
{
    internal abstract class YoloDetectCoreOrtVal<TDetectionResult, TDetectionBatchResult> : YoloDetectCoreBase<TDetectionResult, TDetectionBatchResult>,
        IYoloDetectCore<TDetectionResult, TDetectionBatchResult>
        where TDetectionResult : IYoloSummary<TDetectionResult>
        where TDetectionBatchResult : class, IBatchResultInit<TDetectionResult>, IBatchResultItems<TDetectionResult>, new()
    {
        public YoloDetectCoreOrtVal(InferenceSession session, IDetCorePostprocess<TDetectionResult> postprocess, IDetPreprocess preprocess, OnnxModel onnxModel, YoloConfig config)
          : base(session, postprocess, preprocess, onnxModel, config)
        {

            Warmup();
        }

        protected override void DisposedSub()
        {
        }

        private void Warmup()
        {
            using var outputs = _session.Run(_runOptions, _session.InputNames, [_inputOrtValue], _session.OutputNames);
        }

        public List<TDetectionResult> Run(Mat inputImage)
        {
            // 预处理图像
            var preRes = _preprocess.PreprocessImage(inputImage, _resizedImg, _inputFixedBuffer);

            // 执行推理
            using var outputs = _session.Run(_runOptions, _session.InputNames, [_inputOrtValue], _session.OutputNames);
            using var outputs0 = outputs[0];

            // 后处理
            return _postprocess.PostProcessSync(outputs0, preRes);
        }

        public YoloResult<TDetectionResult> RunWithTime(Mat inputImage)
        {
            SpeedResult speed = new SpeedResult();
            // 预处理图像
            var preRes = PreprocessTime(inputImage, speed);

            _stopwatch.Restart();
            // 执行推理
            using var outputs = _session.Run(_runOptions, _session.InputNames, [_inputOrtValue], _session.OutputNames);
            using var outputs0 = outputs[0];
            _stopwatch.Stop();
            speed.Inference = _stopwatch.ElapsedMilliseconds;

            // 后处理
            var res = PostProcessTime(outputs0, preRes, speed);
            return new YoloResult<TDetectionResult>(res, speed);
        }

        protected override OrtValue RunInferenceBatch(PreDetectResultBatch preResult)
        {
            var outputs = _session.Run(_runOptions, _session.InputNames, [preResult.Data.InputOrtValue], _session.OutputNames);
            _matPool.Return(preResult.Data);
            return outputs[0];
        }
    }
}
