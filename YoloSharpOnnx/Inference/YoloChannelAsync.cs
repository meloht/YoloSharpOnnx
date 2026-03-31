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
            _yoloDetectAsync.InitBufferPool(yoloConfig.BatchPoolSize);
            _concurrentDict = new ConcurrentDictionary<string, TaskCompletionSource<List<DetectionResult>>>();
            var ChannelOptions = GetChannelOptions(yoloConfig.BatchPoolSize);
            _channel = Channel.CreateBounded<PreChannelModel>(ChannelOptions);


            _ = Task.Run(() => ExecuteInferAsync());
        }

        public async Task<List<DetectionResult>> RunDetectAsync(string inputImage)
        {
            YoloValidation.ValidationImagePath(inputImage, _yoloConfig);
            string guid = Guid.NewGuid().ToString();

            if (_yoloDetectAsync.BufferPoolUsedCount >= _yoloConfig.BatchPoolSize)
            {
                await WritePreprocessAsync(inputImage, guid);
            }
            else 
            {
                _ = Task.Run(async () => await WritePreprocessAsync(inputImage, guid));
            }

           
            return await CreateTaskCompletionSource(guid);
        }

        public async Task<List<DetectionResult>> RunDetectAsync(Mat img)
        {
            string guid = Guid.NewGuid().ToString();
            if (_yoloDetectAsync.BufferPoolUsedCount >= _yoloConfig.BatchPoolSize)
            {
                await WritePreprocessAsync(img, guid);
            }
            else
            {
                _ = Task.Run(async () => await WritePreprocessAsync(img, guid));
            }
           
            return await CreateTaskCompletionSource(guid);
        }


        private async ValueTask WritePreprocessAsync(string inputImage, string guid)
        {
            var preResult = _yoloDetectAsync.PreprocessImageChannel(inputImage, _yoloConfig.ResizeAlgorithm);
            await _channel.Writer.WriteAsync(new PreChannelModel(preResult, guid));
        }

        private async ValueTask WritePreprocessAsync(Mat img, string guid)
        {
            var preResult = _yoloDetectAsync.PreprocessImageChannel(img, null, _yoloConfig.ResizeAlgorithm);
            await _channel.Writer.WriteAsync(new PreChannelModel(preResult, guid));
        }

        private async Task<List<DetectionResult>> CreateTaskCompletionSource(string guid)
        {
            var tcs = new TaskCompletionSource<List<DetectionResult>>(TaskCreationOptions.RunContinuationsAsynchronously);
            var ct = new CancellationTokenSource(3000);
            _concurrentDict.TryAdd(guid, tcs);
            ct.Token.Register(() => tcs.TrySetCanceled(), useSynchronizationContext: false);
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
