using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using System;
using System.Buffers;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Channels;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Models;

namespace YoloSharpOnnx.Inference
{
    public class YoloDetectIoBinding : YoloDetectBase, IYoloDetect, IBatchDetect
    {
        private OrtIoBinding _binding;
        protected OrtValue _outputOrtValue;

        public YoloDetectIoBinding(InferenceSession session, SessionOptions options, IPostprocess postprocess, OnnxModel onnxModel)
          : base(session, options, postprocess, onnxModel)
        {

            var outputSizeInBytes = _onnxModel.OutputShapeSize * sizeof(float);
            _binding = _session.CreateIoBinding();

            _outputOrtValue = OrtValue.CreateTensorValueWithData(OrtMemoryInfo.DefaultInstance, TensorElementType.Float,
          _onnxModel.OutputShape, _outputFixedBuffer.Address, outputSizeInBytes);
        }



        public void Dispose()
        {
            DisposeBase();

            _binding.Dispose();
            _outputOrtValue.Dispose();

        }

        public List<DetectionResult> Run(Mat inputImage, YoloConfiguration yoloConfig)
        {
            // 预处理图像
            var preRes = PreprocessImage(inputImage, _resizedImg, _inputFixedBuffer, yoloConfig.ResizeAlgorithm);

            _binding.BindInput(_onnxModel.InputName, _inputOrtValue);
            _binding.BindOutput(_onnxModel.OutputName, _outputOrtValue);
            _binding.SynchronizeBoundInputs();

            // 执行推理

            _session.RunWithBinding(_runOptions, _binding);
            _binding.SynchronizeBoundOutputs();
            // 后处理
            var result = _postprocess.PostProcess(_outputOrtValue, preRes, yoloConfig);
            return result;

        }

        public YoloResult<DetectionResult> RunWithTime(Mat inputImage, YoloConfiguration yoloConfig)
        {
            SpeedResult speed = new SpeedResult();

            _stopwatch.Restart();
            // 预处理图像
            var preRes = PreprocessImage(inputImage, _resizedImg, _inputFixedBuffer, yoloConfig.ResizeAlgorithm);

            _stopwatch.Stop();
            speed.Preprocess = _stopwatch.ElapsedMilliseconds;
            _stopwatch.Restart();

            _binding.BindInput(_onnxModel.InputName, _inputOrtValue);
            _binding.BindOutput(_onnxModel.OutputName, _outputOrtValue);
            //_binding.BindOutputToDevice(_onnxModel.OutputName, OrtMemoryInfo.DefaultInstance);
            //_binding.SynchronizeBoundInputs();

            // 执行推理

            _session.RunWithBinding(_runOptions, _binding);
            _binding.SynchronizeBoundOutputs();

            //using var results = _session.RunWithBoundResults(_runOptions, _binding);
            //_binding.SynchronizeBoundOutputs();
            //using var output = results[0];


            _stopwatch.Stop();
            speed.Inference = _stopwatch.ElapsedMilliseconds;
            _stopwatch.Restart();

            // 后处理
            //var res = _postprocess.PostProcess(output, preRes, yoloConfig);
            var res = _postprocess.PostProcess(_outputOrtValue, preRes, yoloConfig);

            _stopwatch.Stop();
            speed.Postprocess = _stopwatch.ElapsedMilliseconds;
            speed.SumTotal();

            return new YoloResult<DetectionResult>(res, speed);
        }



        public List<DetectionResult> RunBatchDetect(PreResultBatch preRes, YoloConfiguration yoloConfig)
        {
            _binding.BindInput(_onnxModel.InputName, preRes.Data.InputOrtValue);
            _binding.BindOutputToDevice(_onnxModel.OutputName, OrtMemoryInfo.DefaultInstance);
            _binding.SynchronizeBoundInputs();

            // 执行推理
            using var results = _session.RunWithBoundResults(_runOptions, _binding);
            _binding.SynchronizeBoundOutputs();
            using var output = results[0];
            _matPool.Return(preRes.Data);
            // 后处理
            var result = _postprocess.PostProcess(output, preRes.PreResult, yoloConfig);

            return result;
        }

        public DetectionBatchResult[] BatchDetect(List<string> listImg, IBatchProcessCallback processCallback, Action<DetectionBatchResult> receiveAction, int batchPoolSize, YoloConfiguration yoloConfig)
        {
            var task = BatchDetectBaseAsync(listImg, processCallback, receiveAction, batchPoolSize, yoloConfig, this);
            return task.GetAwaiter().GetResult();
        }

        public async Task<DetectionBatchResult[]> BatchDetectAsync(List<string> listImg, IBatchProcessCallback processCallback, Action<DetectionBatchResult> receiveAction, int batchPoolSize, YoloConfiguration yoloConfig)
        {
            return await BatchDetectBaseAsync(listImg, processCallback, receiveAction, batchPoolSize, yoloConfig, this);
        }
    }
}
