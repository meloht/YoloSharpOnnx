using Microsoft.ML.OnnxRuntime;
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
    public class YoloDetectOrtVal : YoloDetectBase, IYoloDetect, IBatchDetect
    {

        public YoloDetectOrtVal(InferenceSession session, SessionOptions options, IPostprocess postprocess, OnnxModel onnxModel)
           : base(session, options, postprocess, onnxModel)
        {
        }


        public void Dispose()
        {
            _session.Dispose();
            _options.Dispose();
            _runOptions.Dispose();
        }

        public List<DetectionResult> Run(Mat inputImage, YoloConfiguration yoloConfig)
        {
            // 预处理图像
            var preRes = PreprocessImg(inputImage, _inputBuffer, yoloConfig.ResizeAlgorithm);

            using var inputOrtValue = OrtValue.CreateTensorValueFromMemory(_inputBuffer, _onnxModel.InputShape);
            // 执行推理
            using var outputs = _session.Run(_runOptions, _session.InputNames, [inputOrtValue], _session.OutputNames);
            using var output0 = outputs[0];

            // 后处理
            var result = _postprocess.PostProcess(output0, preRes, yoloConfig);
            return result;
        }

        public YoloResult<DetectionResult> RunWithTime(Mat inputImage, YoloConfiguration yoloConfig)
        {

            SpeedResult speed = new SpeedResult();
            _stopwatch.Restart();

            // 预处理图像
            var preRes = PreprocessImg(inputImage, _inputBuffer, yoloConfig.ResizeAlgorithm);

            _stopwatch.Stop();
            speed.Preprocess = _stopwatch.ElapsedMilliseconds;
            _stopwatch.Restart();

            using var inputOrtValue = OrtValue.CreateTensorValueFromMemory(_inputBuffer, _onnxModel.InputShape);

            // 执行推理
            using var outputs = _session.Run(_runOptions, _session.InputNames, [inputOrtValue], _session.OutputNames);
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

        public List<DetectionResult> RunBatchDetect(PreResultBatch preRes, YoloConfiguration yoloConfig)
        {
            using var inputOrtValue = OrtValue.CreateTensorValueFromMemory(preRes.Data, _onnxModel.InputShape);
            // 执行推理
            using var outputs = _session.Run(_runOptions, _session.InputNames, [inputOrtValue], _session.OutputNames);
            using var output0 = outputs[0];
            ArrayPool<float>.Shared.Return(preRes.Data);
            // 后处理
            var result = _postprocess.PostProcess(output0, preRes.PreResult, yoloConfig);

            return result;
        }

        public DetectionBatchResult[] BatchDetect(List<string> listImg, int batchSize, YoloConfiguration yoloConfig)
        {
            var task = BatchDetectBase(listImg, batchSize, yoloConfig, this);
            return task.GetAwaiter().GetResult();
        }

        public async Task<DetectionBatchResult[]> BatchDetectAsync(List<string> listImg, int batchSize, YoloConfiguration yoloConfig)
        {
            return await BatchDetectBase(listImg, batchSize, yoloConfig, this);
        }

    }
}
