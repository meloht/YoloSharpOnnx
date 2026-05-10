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

namespace YoloSharpOnnx.Inference
{
    public abstract class YoloChannelAsync<TResult, TBatchPreResult, TPreChannelData> : IYoloTaskAsync<TResult> where TPreChannelData : IGuidValue
    {
        // Producer/consumer
        private readonly Channel<TPreChannelData> _channel;
        private readonly IYoloProcessAsync<TBatchPreResult> _yoloProcessAsync;
        protected readonly IRunBatch<TResult, TBatchPreResult> _runBatch;

        private readonly YoloConfig _yoloConfig;

        private ConcurrentDictionary<Guid, TaskCompletionSource<List<TResult>>> _concurrentDict;

        protected abstract TPreChannelData BuildPreChannelData(TBatchPreResult batchPreResult, Guid guid);
        protected abstract List<TResult> RunBatch(TPreChannelData batchPreResult);

        public YoloChannelAsync(YoloConfig yoloConfig,
            IYoloProcessAsync<TBatchPreResult> yoloProcessAsync,
            IRunBatch<TResult, TBatchPreResult> runBatch)
        {
            _runBatch = runBatch;
            _yoloConfig = yoloConfig;
            _yoloProcessAsync = yoloProcessAsync;
            _yoloProcessAsync.InitBufferPool(yoloConfig.BatchPoolSize);

            _concurrentDict = new ConcurrentDictionary<Guid, TaskCompletionSource<List<TResult>>>();
            var ChannelOptions = YoloUtils.GetChannelOptions(yoloConfig.BatchPoolSize);
            _channel = Channel.CreateBounded<TPreChannelData>(ChannelOptions);


            _ = Task.Run(async () => ExecuteInferAsync());
           
        }

        public async Task<List<TResult>> RunAsync(string inputImage)
        {
            YoloValidation.ValidationImagePath(inputImage, _yoloConfig);
            var guid = Guid.NewGuid();

            if (_yoloProcessAsync.BufferPoolUsedCount >= _yoloConfig.BatchPoolSize)
            {
                await WritePreprocessAsync(inputImage, guid);
            }
            else
            {
                _ = WritePreprocessAsync(inputImage, guid);
            }


            return await CreateTaskCompletionSource(guid);
        }

        public async Task<List<TResult>> RunAsync(Mat img)
        {
            var guid = Guid.NewGuid();
            if (_yoloProcessAsync.BufferPoolUsedCount >= _yoloConfig.BatchPoolSize)
            {
                await WritePreprocessAsync(img, guid);
            }
            else
            {
                _ = WritePreprocessAsync(img, guid);
            }

            return await CreateTaskCompletionSource(guid);
        }


        private async ValueTask WritePreprocessAsync(string inputImage, Guid guid)
        {
            var preResult = _yoloProcessAsync.PreprocessImageChannel(inputImage);
            await _channel.Writer.WriteAsync(BuildPreChannelData(preResult, guid));
        }

        private async ValueTask WritePreprocessAsync(Mat img, Guid guid)
        {
            var preResult = _yoloProcessAsync.PreprocessImageChannel(img, null);
            await _channel.Writer.WriteAsync(BuildPreChannelData(preResult, guid));
        }

        private Task<List<TResult>> CreateTaskCompletionSource(Guid guid)
        {
            var tcs = new TaskCompletionSource<List<TResult>>(TaskCreationOptions.RunContinuationsAsynchronously);
            var ct = new CancellationTokenSource(_yoloConfig.AsyncChannelTimeout);
            _concurrentDict.TryAdd(guid, tcs);

            ct.Token.Register(() => tcs.TrySetCanceled(), useSynchronizationContext: false);
            return tcs.Task;
        }


        private async ValueTask ExecuteInferAsync()
        {
            await foreach (TPreChannelData item in _channel.Reader.ReadAllAsync())
            {
                var result = RunBatch(item);

                TaskCompletionSource<List<TResult>> tempTCS = _concurrentDict[item.Guid];
                tempTCS.TrySetResult(result);
                _concurrentDict.TryRemove(item.Guid, out tempTCS);

            }
        }

        public void Dispose()
        {
            _channel.Writer.Complete();
        }





    }
}
