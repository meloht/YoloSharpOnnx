using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Reflection.PortableExecutable;
using System.Text;
using System.Threading.Channels;
using System.Threading.Tasks;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Models;

namespace YoloSharpOnnx.Inference
{
    public class YoloChannelAsync : IYoloAsync
    {
        // Producer/consumer
        private readonly Channel<PreChannelModel> _channel;
        private readonly IYoloDetectAsync _yoloDetectAsync;
        private readonly YoloConfig _yoloConfig;

        private ConcurrentDictionary<string, TaskCompletionSource<List<DetectionResult>>> _concurrentDict;

        public YoloChannelAsync(YoloConfig yoloConfig, IYoloDetectAsync yoloDetectAsync)
        {
            _yoloDetectAsync = yoloDetectAsync;
            _yoloConfig = yoloConfig;
            _concurrentDict = new ConcurrentDictionary<string, TaskCompletionSource<List<DetectionResult>>>();
            var ChannelOptions = GetChannelOptions(yoloConfig.BatchPoolSize);
            _channel = Channel.CreateBounded<PreChannelModel>(ChannelOptions);
            _yoloDetectAsync.InitBufferPool(yoloConfig.BatchPoolSize);

            _ = Task.Run(() => ExecuteInferAsync());
        }

        public async Task<List<DetectionResult>> RunDetectAsync(string inputImage)
        {
            // 预处理图像
            var preResult = _yoloDetectAsync.PreprocessImageChannel(inputImage, _yoloConfig.ResizeAlgorithm);
            return await ComletedInferAsync(preResult);
        }

        public async Task<List<DetectionResult>> RunDetectAsync(Mat img)
        {
            // 预处理图像
            var preResult = _yoloDetectAsync.PreprocessImageChannel(img, null, _yoloConfig.ResizeAlgorithm);
            return await ComletedInferAsync(preResult);
        }
        private async Task<List<DetectionResult>> ComletedInferAsync(PreResultBatch preResult)
        {
            string guid = Guid.NewGuid().ToString();

            var tcs = new TaskCompletionSource<List<DetectionResult>>(TaskCreationOptions.RunContinuationsAsynchronously);
            var ct = new CancellationTokenSource(1000);
            _concurrentDict.TryAdd(guid, tcs);
            ct.Token.Register(() => tcs.TrySetCanceled(), useSynchronizationContext: false);

            await _channel.Writer.WriteAsync(new PreChannelModel(preResult, guid));

            return await tcs.Task;
        }


        private async ValueTask ExecuteInferAsync()
        {
            await foreach (PreChannelModel item in _channel.Reader.ReadAllAsync())
            {
                var result = _yoloDetectAsync.RunBatchDetect(item.PreResult, _yoloConfig);

                var tempTCS = _concurrentDict[item.Guid];
                tempTCS.SetResult(result);
                _concurrentDict.TryRemove(item.Guid, out tempTCS);

            }
        }

        public void Dispose()
        {
            _channel.Writer.Complete();
        }

        private BoundedChannelOptions GetChannelOptions(int batchPoolSize)
        {
            var channelOptions = new BoundedChannelOptions(batchPoolSize)
            {
                SingleWriter = false,
                SingleReader = true,
                AllowSynchronousContinuations = false,
                FullMode = BoundedChannelFullMode.Wait
            };

            return channelOptions;
        }



    }
}
