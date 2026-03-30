using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Reflection.PortableExecutable;
using System.Text;
using System.Threading.Channels;
using System.Threading.Tasks;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Models;

namespace YoloSharpOnnx.Inference
{
    public class YoloDetectOrtValAsync : IDisposable
    {
        private readonly OnnxModel _onnxModel;
        // Producer/consumer
        private readonly Channel<PreResultBatch> _channel;
        private IPostprocess _postprocess;
        private IPreprocess _preprocess;
        private MatBufferPool _matBufferPool;
        private InferenceSession _session;

        protected SessionOptions _options;
        protected RunOptions _runOptions;


        private ConcurrentDictionary<string, TaskCompletionSource<List<DetectionResult>>> _concurrentDictionary;

        public YoloDetectOrtValAsync(OnnxModel onnxModel, YoloConfig yoloConfig)
        {
            _onnxModel = onnxModel;
            _concurrentDictionary = new ConcurrentDictionary<string, TaskCompletionSource<List<DetectionResult>>>();
            var ChannelOptions = GetChannelOptions(yoloConfig.BatchPoolSize);
            _channel = Channel.CreateBounded<PreResultBatch>(ChannelOptions);

        }

        public List<DetectionResult> RunAsync(string inputImage, YoloConfig yoloConfig)
        {
            // 预处理图像

            var producer = RunPreprocessAsync(inputImage, yoloConfig.ResizeAlgorithm);

            //var consumer = Task.Run(async () =>
            //{

            //    await foreach (PreResultBatch item in _channel.Reader.ReadAllAsync())
            //    {
            //        long startTime = DateTimeOffset.Now.ToUnixTimeMilliseconds();
            //        var result = batchDetect.RunBatchDetect(item, yoloConfig);
            //        var modelResult = new DetectionBatchResult(item.ImagePath, result, startTime);
            //        batchResults[idx] = modelResult;
            //        Interlocked.Increment(ref idx);
            //        await InferCompleteAsync(modelResult, processCallback, receiveAction);
            //    }
            //});
            //await Task.WhenAll(producer, consumer);

            //// 执行推理
            //using var outputs = _session.Run(_runOptions, _session.InputNames, [_inputOrtValue], _session.OutputNames);
            //using var output0 = outputs[0];

            //// 后处理
            //var result = _postprocess.PostProcess(output0, preRes, yoloConfig);
            //return result;
            return null;
        }

        public YoloResult<DetectionResult> RunWithTimeAsync(Mat inputImage, YoloConfig yoloConfig)
        {
            return null;
        }

        private void InferCompleteAsync()
        {
            var consumer = Task.Run(async () =>
            {

                await foreach (PreResultBatch item in _channel.Reader.ReadAllAsync())
                {
                    long startTime = DateTimeOffset.Now.ToUnixTimeMilliseconds();
                   // var result = batchDetect.RunBatchDetect(item, yoloConfig);
                    //var modelResult = new DetectionBatchResult(item.ImagePath, result, startTime);
                   // batchResults[idx] = modelResult;
                   // Interlocked.Increment(ref idx);
                    //await InferCompleteAsync(modelResult, processCallback, receiveAction);
                }
            });
           
            Task.WaitAll(consumer);
        }

        private async Task RunPreprocessAsync(string imagePath, InterpolationFlags interpolationFlags)
        {
            await Task.Run(async () =>
            {
                var data = _matBufferPool.Rent();
                using Mat img = Cv2.ImRead(imagePath);
                var res = _preprocess.PreprocessImage(img, data.ResizedImg, data.FixedBuffer, interpolationFlags);
                await _channel.Writer.WriteAsync(new PreResultBatch(res, imagePath, data));
            });
        }

        public void Dispose()
        {
            _channel.Writer.Complete();
        }

        private BoundedChannelOptions GetChannelOptions(int batchPoolSize)
        {
            var channelOptions = new BoundedChannelOptions(batchPoolSize)
            {
                SingleWriter = true,
                SingleReader = true,
                AllowSynchronousContinuations = false,
                FullMode = BoundedChannelFullMode.Wait
            };

            return channelOptions;
        }



    }
}
