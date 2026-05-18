using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using System;
using System.Buffers;
using System.Collections.Generic;
using System.Text;
using System.Threading.Channels;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Inference.Classify.Models;
using YoloSharpOnnx.Inference.Detect.Models;
using YoloSharpOnnx.Inference.Segment.Models;
using YoloSharpOnnx.Models;

namespace YoloSharpOnnx.Inference.Detect
{
    public class YoloDetectOrtVal : YoloDetectBase, IYoloDetect
    {
        public YoloDetectOrtVal(InferenceSession session, SessionOptions options, IDetPostprocess postprocess, IDetPreprocess preprocess, OnnxModel onnxModel, YoloConfig config)
           : base(session, options, postprocess, preprocess, onnxModel, config)
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

        public List<DetectionResult> Run(Mat inputImage)
        {
            // 预处理图像
            var preRes = _preprocess.PreprocessImage(inputImage, _inputFixedBuffer);

            // 执行推理
            using var outputs = _session.Run(_runOptions, _session.InputNames, [_inputOrtValue], _session.OutputNames);
            using var outputs0 = outputs[0];

            // 后处理
            return _postprocess.PostProcess(outputs0, preRes);
        }

        public YoloResult<DetectionResult> RunWithTime(Mat inputImage)
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
            return new YoloResult<DetectionResult>(res, speed);
        }

        protected override OrtValue RunInferenceBatch(PreDetectResultBatch preResult)
        {
            var outputs = _session.Run(_runOptions, _session.InputNames, [preResult.Data.InputOrtValue], _session.OutputNames);
            _matPool.Return(preResult.Data);
            return outputs[0];
        }

    }
}
