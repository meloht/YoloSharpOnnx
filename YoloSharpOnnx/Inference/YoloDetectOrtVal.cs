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

namespace YoloSharpOnnx.Inference
{
    public class YoloDetectOrtVal : YoloDetectBase, IYoloDetect, IYoloDetectAsync
    {

        public YoloDetectOrtVal(InferenceSession session, SessionOptions options, IPostprocess postprocess, IPreprocess preprocess, OnnxModel onnxModel)
           : base(session, options, postprocess, preprocess, onnxModel)
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
        public List<DetectionResult> Run(Mat inputImage, YoloConfig yoloConfig)
        {
            // 预处理图像
            var preRes = _preprocess.PreprocessImage(inputImage, _resizedImg, _inputFixedBuffer, yoloConfig.ResizeAlgorithm);

            // 执行推理
            using var outputs = _session.Run(_runOptions, _session.InputNames, [_inputOrtValue], _session.OutputNames);
            using var output0 = outputs[0];

            // 后处理
            var result = _postprocess.PostProcess(output0, preRes, yoloConfig);
            return result;
        }

        public YoloResult<DetectionResult> RunWithTime(Mat inputImage, YoloConfig yoloConfig)
        {

            SpeedResult speed = new SpeedResult();
            _stopwatch.Restart();

            // 预处理图像
            var preRes = _preprocess.PreprocessImage(inputImage, _resizedImg, _inputFixedBuffer, yoloConfig.ResizeAlgorithm);

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
            var res = _postprocess.PostProcess(output0, preRes, yoloConfig);

            _stopwatch.Stop();
            speed.Postprocess = _stopwatch.ElapsedMilliseconds;
            speed.SumTotal();

            return new YoloResult<DetectionResult>(res, speed);
        }

        public List<DetectionResult> RunBatchDetect(PreResultBatch preRes, YoloConfig yoloConfig)
        {
            // 执行推理
            using var outputs = _session.Run(_runOptions, _session.InputNames, [preRes.Data.InputOrtValue], _session.OutputNames);
            using var output0 = outputs[0];
            _matPool.Return(preRes.Data);
            // 后处理
            var result = _postprocess.PostProcess(output0, preRes.PreResult, yoloConfig);

            return result;
        }

        public DetectionBatchResult[] BatchDetect(List<string> listImg, IBatchProcessCallback processCallback, Action<DetectionBatchResult> receiveAction, YoloConfig yoloConfig)
        {
            var task = BatchDetectBaseAsync(listImg, processCallback, receiveAction, yoloConfig, this);
            return task.GetAwaiter().GetResult();
        }

        public async Task<DetectionBatchResult[]> BatchDetectAsync(List<string> listImg, IBatchProcessCallback processCallback, Action<DetectionBatchResult> receiveAction, YoloConfig yoloConfig)
        {
            return await BatchDetectBaseAsync(listImg, processCallback, receiveAction, yoloConfig, this);
        }


        public IYoloDetectAsync GetYoloDetectAsync()
        {
            return this;
        }

        public IAsyncEnumerable<DetectionBatchResult> BatchDetectForeachAsync(List<string> listImg, YoloConfig yoloConfig)
        {
            return BatchDetectBaseForeachAsync(listImg, yoloConfig, this);
        }
    }
}
