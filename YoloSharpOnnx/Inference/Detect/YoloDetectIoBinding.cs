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
using YoloSharpOnnx.Inference.Detect.Models;
using YoloSharpOnnx.Models;

namespace YoloSharpOnnx.Inference.Detect
{
    public class YoloDetectIoBinding : YoloDetectBase, IYoloDetect
    {
        private readonly OrtIoBinding _binding;
        private readonly OrtValue _outputOrtValue;
        private readonly FixedBuffer _outputFixedBuffer;

        public YoloDetectIoBinding(InferenceSession session, SessionOptions options, IDetPostprocess postprocess, IDetPreprocess preprocess, OnnxModel onnxModel, YoloConfig config)
          : base(session, options, postprocess, preprocess, onnxModel, config)
        {

            _binding = _session.CreateIoBinding();
            _outputFixedBuffer = new FixedBuffer(_onnxModel.OutputShapeSize0);
            _outputOrtValue = OrtValue.CreateTensorValueWithData(OrtMemoryInfo.DefaultInstance, TensorElementType.Float,
            _onnxModel.OutputShape0, _outputFixedBuffer.Address, _onnxModel.OutputSizeInBytes0);

            RunInference();
        }

        protected override void DisposedSub()
        {
            _binding.Dispose();
            _outputOrtValue.Dispose();
            _outputFixedBuffer.Dispose();
        }


        private void RunInference()
        {
            _binding.BindInput(_onnxModel.InputName, _inputOrtValue);
            _binding.BindOutput(_onnxModel.OutputName0, _outputOrtValue);
            _binding.SynchronizeBoundInputs();

            //_binding.BindOutputToDevice(_onnxModel.OutputName, OrtMemoryInfo.DefaultInstance);
            //_binding.SynchronizeBoundInputs();

            // 执行推理

            _session.RunWithBinding(_runOptions, _binding);
            _binding.SynchronizeBoundOutputs();

            //using var results = _session.RunWithBoundResults(_runOptions, _binding);
            //_binding.SynchronizeBoundOutputs();
            //using var output = results[0];

        }
        public List<DetectionResult> Run(Mat inputImage)
        {
            // 预处理图像
            var preRes = _preprocess.PreprocessImage(inputImage, _resizedImg, _inputFixedBuffer);

            // 执行推理
            RunInference();

            // 后处理
            return _postprocess.PostProcess(_outputOrtValue, preRes);
        }

        public YoloResult<DetectionResult> RunWithTime(Mat inputImage)
        {
            SpeedResult speed = new SpeedResult();
            // 预处理图像
            var preRes = PreprocessTime(inputImage, speed);

            _stopwatch.Restart();
            // 执行推理
            RunInference();
            _stopwatch.Stop();
            speed.Inference = _stopwatch.ElapsedMilliseconds;

            // 后处理
            var res = PostProcessTime(_outputOrtValue, preRes, speed);
            return new YoloResult<DetectionResult>(res, speed);
        }



        protected override List<DetectionResult> RunBatchInfer(PreDetectResultBatch preResult)
        {
            bool isReturn = false;
            try
            {
                _binding.BindInput(_onnxModel.InputName, preResult.Data.InputOrtValue);
                _binding.BindOutputToDevice(_onnxModel.OutputName0, OrtMemoryInfo.DefaultInstance);
                _binding.SynchronizeBoundInputs();

                // 执行推理
                using var results = _session.RunWithBoundResults(_runOptions, _binding);
                _binding.SynchronizeBoundOutputs();
                using var output = results[0];
                _matPool.Return(preResult.Data);
                isReturn = true;
                // 后处理
                var result = _postprocess.PostProcess(output, preResult.PreResult);

                return result;
            }
            finally
            {
                if (!isReturn)
                {
                    _matPool.Return(preResult.Data);
                }
            }

        }

        protected override Task RunBatchInfer(DetectionBatchResult[] batchResults, int idx, PreDetectResultBatch item, long startTime, IBatchProcessCallback<DetectionBatchResult> processCallback, Action<DetectionBatchResult> receiveAction)
        {
            bool isReturn = false;
            try
            {
                _binding.BindInput(_onnxModel.InputName, item.Data.InputOrtValue);
                _binding.BindOutputToDevice(_onnxModel.OutputName0, OrtMemoryInfo.DefaultInstance);
                _binding.SynchronizeBoundInputs();

                // 执行推理
                var results = _session.RunWithBoundResults(_runOptions, _binding);
                _binding.SynchronizeBoundOutputs();

                _matPool.Return(item.Data);
                isReturn = true;

                // 后处理
                return BatchPostProcess(batchResults, idx, results[0], item, startTime, processCallback, receiveAction);
            }
            finally
            {
                if (!isReturn)
                {
                    _matPool.Return(item.Data);
                }
            }
        }
    }
}
