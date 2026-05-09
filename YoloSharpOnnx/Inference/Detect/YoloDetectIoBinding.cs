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
    public class YoloDetectIoBinding : YoloDetectBase, IYoloDetect, IYoloDetectAsync, IBatchProcess<DetectionResult, PreDetectResultBatch, DetectionBatchResult>
    {
        private OrtIoBinding _binding;
        protected OrtValue _outputOrtValue;
        private bool disposedValue;

        public YoloDetectIoBinding(InferenceSession session, SessionOptions options, IDetPostprocess postprocess, IDetPreprocess preprocess, OnnxModel onnxModel, YoloConfig config)
          : base(session, options, postprocess, preprocess, onnxModel, config)
        {

            _binding = _session.CreateIoBinding();

            _outputOrtValue = OrtValue.CreateTensorValueWithData(OrtMemoryInfo.DefaultInstance, TensorElementType.Float,
          _onnxModel.OutputShape, _outputFixedBuffer.Address, _onnxModel.OutputSizeInBytes);

            InitBatchProcess(this);

            Warmup();
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!disposedValue)
            {
                if (disposing)
                {
                    // TODO: dispose managed state (managed objects)
                }

                // TODO: free unmanaged resources (unmanaged objects) and override finalizer
                // TODO: set large fields to null
                DisposeCore();

                _binding.Dispose();
                _outputOrtValue.Dispose();
                disposedValue = true;
            }
        }

        // // TODO: override finalizer only if 'Dispose(bool disposing)' has code to free unmanaged resources
        // ~YoloDetectIoBinding()
        // {
        //     // Do not change this code. Put cleanup code in 'Dispose(bool disposing)' method
        //     Dispose(disposing: false);
        // }

        public void Dispose()
        {
            // Do not change this code. Put cleanup code in 'Dispose(bool disposing)' method
            Dispose(disposing: true);
            GC.SuppressFinalize(this);
        }

        private void Warmup()
        {
            _binding.BindInput(_onnxModel.InputName, _inputOrtValue);
            _binding.BindOutput(_onnxModel.OutputName, _outputOrtValue);
            _binding.SynchronizeBoundInputs();

            _session.RunWithBinding(_runOptions, _binding);
            _binding.SynchronizeBoundOutputs();
        }
        public List<DetectionResult> Run(Mat inputImage)
        {
            // 预处理图像
            var preRes = _preprocess.PreprocessImage(inputImage, _resizedImg, _inputFixedBuffer, _config.ResizeAlgorithm);

            _binding.BindInput(_onnxModel.InputName, _inputOrtValue);
            _binding.BindOutput(_onnxModel.OutputName, _outputOrtValue);
            _binding.SynchronizeBoundInputs();

            // 执行推理

            _session.RunWithBinding(_runOptions, _binding);
            _binding.SynchronizeBoundOutputs();
            // 后处理
            var result = _postprocess.PostProcess(_outputOrtValue, preRes, _config);
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
            var res = _postprocess.PostProcess(_outputOrtValue, preRes, _config);

            _stopwatch.Stop();
            speed.Postprocess = _stopwatch.ElapsedMilliseconds;
            speed.SumTotal();

            return new YoloResult<DetectionResult>(res, speed);
        }



        public List<DetectionResult> RunBatch(PreDetectResultBatch preRes)
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
            var result = _postprocess.PostProcess(output, preRes.PreResult, _config);

            return result;
        }

        public DetectionBatchResult[] BatchDetect(List<string> listImg, IBatchProcessCallback<DetectionBatchResult> processCallback, Action<DetectionBatchResult> receiveAction)
        {
            return BatchDetectBase(listImg, processCallback, receiveAction); 
        }

        public async Task<DetectionBatchResult[]> BatchDetectAsync(List<string> listImg, IBatchProcessCallback<DetectionBatchResult> processCallback, Action<DetectionBatchResult> receiveAction)
        {
            return await BatchDetectBaseAsync(listImg, processCallback, receiveAction);
        }

        public IYoloDetectAsync GetYoloDetectAsync()
        {
            return this;
        }

        public IAsyncEnumerable<DetectionBatchResult> BatchDetectForeachAsync(List<string> listImg)
        {
            return BatchDetectBaseForeachAsync(listImg);
        }

        public PreDetectResultBatch GetPreprocessImageBatchData(Mat inputImage, ImageBatchData imageBatchData, string imagePath)
        {
            var preRes = _preprocess.PreprocessImage(inputImage, imageBatchData.ResizedImg, imageBatchData.FixedBuffer, _config.ResizeAlgorithm);
            return new PreDetectResultBatch(preRes, imagePath, imageBatchData);
        }

        public DetectionBatchResult BuildBatchResult(PreDetectResultBatch batchPreResult, List<DetectionResult> results, long timestamp)
        {
            return new DetectionBatchResult(batchPreResult.ImagePath, results, timestamp);
        }
        public IRunBatch<DetectionResult, PreDetectResultBatch> GetRunBatch()
        {
            return this;
        }



       
    }
}
