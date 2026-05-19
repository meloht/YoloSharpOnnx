using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Channels;
using System.Threading.Tasks;

namespace YoloSharpOnnx.Inference
{
    public class YoloChannelAsync<TResult, TBatchPreResult, TPreChannelData> : IYoloTaskAsync<TResult> where TPreChannelData : class, IGuidValue<TBatchPreResult>, IBatchPreChannelResult<TBatchPreResult>, new()
    {
        // Producer/consumer
        private readonly Channel<TPreChannelData> _channel;
        private readonly IYoloProcessAsync<TBatchPreResult> _yoloProcessAsync;
        protected readonly IRunBatch<TResult, TBatchPreResult> _runBatch;

        private readonly YoloConfig _yoloConfig;

        private ConcurrentDictionary<Guid, TaskCompletionSource<List<TResult>>> _concurrentDict;

        protected readonly Lazy<ObjectPool<TPreChannelData>> _preChannelPool;

 
        public YoloChannelAsync(YoloConfig yoloConfig,
            IYoloProcessAsync<TBatchPreResult> yoloProcessAsync,
            IRunBatch<TResult, TBatchPreResult> runBatch)
        {
            _runBatch = runBatch;
            _yoloConfig = yoloConfig;
            _yoloProcessAsync = yoloProcessAsync;
            _yoloProcessAsync.InitBufferPool(yoloConfig.BatchPoolSize);

            _preChannelPool = new Lazy<ObjectPool<TPreChannelData>>(() => new ObjectPool<TPreChannelData>(() => new TPreChannelData(), yoloConfig.BatchPoolSize));

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

        private TPreChannelData BuildPreChannelData(TBatchPreResult preResult,Guid guid)
        {
            var data = _preChannelPool.Value.Rent();
            data.Initialize(guid, preResult);
            return data;
        }

        private async ValueTask ExecuteInferAsync()
        {
            await foreach (TPreChannelData item in _channel.Reader.ReadAllAsync())
            {
                try
                {
                    var result = _runBatch.RunBatch(item.PreResult);

                    TaskCompletionSource<List<TResult>> tempTCS = _concurrentDict[item.Guid];
                    tempTCS.TrySetResult(result);
                    _concurrentDict.TryRemove(item.Guid, out tempTCS);
                }
                finally
                {
                    _preChannelPool.Value.Return(item);
                    _runBatch.ReturnBatchPreResult(item.PreResult);
                }
            }
        }

        public void Dispose()
        {
            _channel.Writer.Complete();
            if (_preChannelPool.IsValueCreated)
            {
                _preChannelPool.Value.Dispose();
            }
        }
    }
}
