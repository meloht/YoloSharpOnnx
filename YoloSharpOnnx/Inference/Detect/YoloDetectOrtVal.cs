using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using System;
using System.Buffers;
using System.Collections.Generic;
using System.Text;
using System.Threading.Channels;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Models;

namespace YoloSharpOnnx.Inference.Detect
{
    public class YoloDetectOrtVal : YoloDetectBase, IYoloDetect, IYoloDetectAsync
    {

        public YoloDetectOrtVal(InferenceSession session, SessionOptions options, IDetPostprocess postprocess, IDetPreprocess preprocess, OnnxModel onnxModel, YoloConfig config)
           : base(session, options, postprocess, preprocess, onnxModel, config)
        {
            Warmup();
        }


        public void Dispose()
        {
            DisposeBase();
        }
        private void Warmup()
        {
            using var outputs = _session.Run(_runOptions, _session.InputNames, [_inputOrtValue], _session.OutputNames);
            using var output0 = outputs[0];
        }
        public List<DetectionResult> Run(Mat inputImage)
        {
            // 预处理图像
            var preRes = _preprocess.PreprocessImage(inputImage, _resizedImg, _inputFixedBuffer, _config.ResizeAlgorithm);

            // 执行推理
            using var outputs = _session.Run(_runOptions, _session.InputNames, [_inputOrtValue], _session.OutputNames);
            using var output0 = outputs[0];

            // 后处理
            var result = _postprocess.PostProcess(output0, preRes, _config);
            return result;
        }

        public YoloResult<DetectionResult> RunWithTime(Mat inputImage)
        {

            SpeedResult speed = new SpeedResult();
            _stopwatch.Restart();

            // 预处理图像
            var preRes = _preprocess.PreprocessImage(inputImage, _resizedImg, _inputFixedBuffer, _config.ResizeAlgorithm);

            _stopwatch.Stop();
            speed.Preprocess = _stopwatch.ElapsedMilliseconds;
            _stopwatch.Restart();

            // 执行推理
            using var outputs = _session.Run(_runOptions, _session.InputNames, [_inputOrtValue], _session.OutputNames);
            using var output0 = outputs[0];

            _stopwatch.Stop();
            speed.Inference = _stopwatch.ElapsedMilliseconds;
            _stopwatch.Restart();


            // 后处理
            var res = _postprocess.PostProcess(output0, preRes, _config);

            _stopwatch.Stop();
            speed.Postprocess = _stopwatch.ElapsedMilliseconds;
            speed.SumTotal();

            return new YoloResult<DetectionResult>(res, speed);
        }

        public List<DetectionResult> RunBatchDetect(PreResultBatch preRes)
        {
            // 执行推理
            using var outputs = _session.Run(_runOptions, _session.InputNames, [preRes.Data.InputOrtValue], _session.OutputNames);
            using var output0 = outputs[0];
            _matPool.Return(preRes.Data);
            // 后处理
            var result = _postprocess.PostProcess(output0, preRes.PreResult, _config);

            return result;
        }

        public DetectionBatchResult[] BatchDetect(List<string> listImg, IBatchProcessCallback processCallback, Action<DetectionBatchResult> receiveAction)
        {
            var task = BatchDetectBaseAsync(listImg, processCallback, receiveAction, this);
            return task.GetAwaiter().GetResult();
        }

        public async Task<DetectionBatchResult[]> BatchDetectAsync(List<string> listImg, IBatchProcessCallback processCallback, Action<DetectionBatchResult> receiveAction)
        {
            return await BatchDetectBaseAsync(listImg, processCallback, receiveAction, this);
        }


        public IYoloDetectAsync GetYoloDetectAsync()
        {
            return this;
        }

        public IAsyncEnumerable<DetectionBatchResult> BatchDetectForeachAsync(List<string> listImg)
        {
            return BatchDetectBaseForeachAsync(listImg, this);
        }
    }
}
