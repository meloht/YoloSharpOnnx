using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Inference.Detect;
using YoloSharpOnnx.Inference.Detect.Models;
using YoloSharpOnnx.Models;

namespace YoloSharpOnnx.Inference.Segment
{
    public class YoloSegOrtVal : YoloSegBase, IYoloSegment
    {
        public YoloSegOrtVal(InferenceSession session, SessionOptions options, ISegPostprocess postprocess, IDetPreprocess preprocess, OnnxModel onnxModel, YoloConfig config)
            : base(session, options, postprocess, preprocess, onnxModel, config)
        {
            Warmup();
        }
        private void Warmup()
        {
            using var outputs = _session.Run(_runOptions, _session.InputNames, [_inputOrtValue], _session.OutputNames);

        }
        public List<SegResult> Run(Mat inputImage)
        {
            // 预处理图像
            var preRes = _preprocess.PreprocessImage(inputImage, _resizedImg, _inputFixedBuffer);

            // 执行推理
            using var outputs = _session.Run(_runOptions, _session.InputNames, [_inputOrtValue], _session.OutputNames);
            using var output0 = outputs[0];
            using var output1 = outputs[1];

            // 后处理
            return _postprocess.PostProcess(output0, output1, preRes);
        }

        public YoloResult<SegResult> RunWithTime(Mat inputImage)
        {
            SpeedResult speed = new SpeedResult();
            // 预处理图像
            var preRes = PreprocessTime(inputImage, speed);

            _stopwatch.Restart();
            // 执行推理
            using var outputs = _session.Run(_runOptions, _session.InputNames, [_inputOrtValue], _session.OutputNames);
            using var output0 = outputs[0];
            using var output1 = outputs[1];
            _stopwatch.Stop();
            speed.Inference = _stopwatch.ElapsedMilliseconds;

            // 后处理
            var res = PostProcessTime(output0, output1, preRes, speed);
            return new YoloResult<SegResult>(res, speed);
        }

        protected override void DisposedSub()
        {

        }


        protected override IDisposableReadOnlyCollection<OrtValue> RunInferenceBatch(PreDetectResultBatch preResult)
        {
            var outputs = _session.Run(_runOptions, _session.InputNames, [preResult.Data.InputOrtValue], _session.OutputNames);
            return outputs;
        }
    }
}
