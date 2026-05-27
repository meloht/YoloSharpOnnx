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
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Inference.Detect.Models;

namespace YoloSharpOnnx.Inference
{
    internal class YoloChannelAsync<TResult, TBatchPreResult, TAsyncResult> : IYoloTaskAsync<TResult, TAsyncResult>
        where TBatchPreResult : class
        where TAsyncResult : class, IChannelAsyncResult<TResult>, new()
    {
        // Producer/consumer
        private readonly Channel<PreDetectChannelData<TAsyncResult, TBatchPreResult>> _channel;
        private readonly IYoloProcessAsync<TBatchPreResult, TResult> _yoloProcessAsync;
        private readonly YoloConfig _yoloConfig;

        private ConcurrentDictionary<Guid, TaskCompletionSource<List<TResult>>> _concurrentDict;

        protected readonly Lazy<ObjectPool<PreDetectChannelData<TAsyncResult, TBatchPreResult>>> _preChannelPool;

        private ValueTask _inferTask;
        private volatile bool _channelClosed = false;

        public YoloChannelAsync(YoloConfig yoloConfig, IYoloProcessAsync<TBatchPreResult, TResult> yoloProcessAsync)
        {

            _yoloConfig = yoloConfig;
            _yoloProcessAsync = yoloProcessAsync;
            _yoloProcessAsync.InitBufferPool(yoloConfig.BatchPoolSize);

            _preChannelPool = new Lazy<ObjectPool<PreDetectChannelData<TAsyncResult, TBatchPreResult>>>(() => new ObjectPool<PreDetectChannelData<TAsyncResult, TBatchPreResult>>(() => new PreDetectChannelData<TAsyncResult, TBatchPreResult>(), yoloConfig.BatchPoolSize));

            _concurrentDict = new ConcurrentDictionary<Guid, TaskCompletionSource<List<TResult>>>();
            var ChannelOptions = YoloUtils.GetChannelOptions(yoloConfig.BatchPoolSize);
            _channel = Channel.CreateBounded<PreDetectChannelData<TAsyncResult, TBatchPreResult>>(ChannelOptions);

            _inferTask = ExecuteInferAsync();

        }

        public async Task CompleteAndCloseAsyncChannel()
        {
            CloseChannel();
            await _inferTask;
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

        public async Task RunAsync(Mat img, Guid guid, IBatchProcessCallback<TAsyncResult> callback, Action<TAsyncResult> receiveAction)
        {
            await WritePreprocessAsync(img, guid, callback, receiveAction);
        }


        private async ValueTask WritePreprocessAsync(string inputImage, Guid guid)
        {
            var preResult = _yoloProcessAsync.PreprocessImageChannel(inputImage);
            await _channel.Writer.WriteAsync(BuildPreChannelData(preResult, guid, null, null));
        }

        private async ValueTask WritePreprocessAsync(Mat img, Guid guid)
        {
            var preResult = _yoloProcessAsync.PreprocessImageChannel(img, null);
            await _channel.Writer.WriteAsync(BuildPreChannelData(preResult, guid, null, null));
        }

        private async ValueTask WritePreprocessAsync(Mat img, Guid guid, IBatchProcessCallback<TAsyncResult> callback, Action<TAsyncResult> receiveAction)
        {
            var preResult = _yoloProcessAsync.PreprocessImageChannel(img, null);
            await _channel.Writer.WriteAsync(BuildPreChannelData(preResult, guid, callback, receiveAction));
        }

        private Task<List<TResult>> CreateTaskCompletionSource(Guid guid)
        {
            var tcs = new TaskCompletionSource<List<TResult>>(TaskCreationOptions.RunContinuationsAsynchronously);
            var ct = new CancellationTokenSource(_yoloConfig.AsyncChannelTimeout);
            _concurrentDict.TryAdd(guid, tcs);

            ct.Token.Register(() => tcs.TrySetCanceled(), useSynchronizationContext: false);
            return tcs.Task;
        }

        private PreDetectChannelData<TAsyncResult, TBatchPreResult> BuildPreChannelData(TBatchPreResult preResult, Guid guid, IBatchProcessCallback<TAsyncResult> callback, Action<TAsyncResult> receiveAction)
        {
            var data = _preChannelPool.Value.Rent();
            data.Initialize(guid, preResult, callback, receiveAction);
            return data;
        }

        private async ValueTask ExecuteInferAsync()
        {
            await foreach (PreDetectChannelData<TAsyncResult, TBatchPreResult> item in _channel.Reader.ReadAllAsync())
            {
                try
                {
                    long startTime = DateTimeOffset.Now.ToUnixTimeMilliseconds();
                    var result = _yoloProcessAsync.RunBatch(item.PreResult);

                    if (_concurrentDict.TryGetValue(item.Guid, out TaskCompletionSource<List<TResult>> tempTCS))
                    {
                        tempTCS.TrySetResult(result);
                        _concurrentDict.TryRemove(item.Guid, out tempTCS);
                    }
                    TAsyncResult asyncResult = new();
                    asyncResult.Initialize(item.Guid, result, startTime);
                    _ = InferCompleteAsync(asyncResult, item.Callback, item.ReceiveAction);
                }
                finally
                {
                    _preChannelPool.Value.Return(item);
                }
            }
        }
        private async Task InferCompleteAsync(TAsyncResult result, IBatchProcessCallback<TAsyncResult> processCallback, Action<TAsyncResult> receiveAction)
        {
            if (processCallback != null)
            {
                await Task.Run(() =>
                {
                    processCallback.ReceiveProcessResult(result);
                });
            }
            if (receiveAction != null)
            {
                await Task.Run(() =>
                {
                    receiveAction(result);
                });
            }
        }
        public void Dispose()
        {
            CloseChannel();

            if (_preChannelPool.IsValueCreated)
            {
                _preChannelPool.Value.Dispose();
            }
        }

        private void CloseChannel()
        {
            if (!_channelClosed)
            {
                _channelClosed = true;
                _channel.Writer.TryComplete();
            }
        }
    }
}
