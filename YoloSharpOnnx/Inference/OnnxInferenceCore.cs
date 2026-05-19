using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Channels;
using System.Threading.Tasks;
using YoloSharpOnnx.Inference.Segment.Models;
using YoloSharpOnnx.Models;

namespace YoloSharpOnnx.Inference
{
    public abstract class OnnxInferenceCore<TResult, TBatchPreResult, TBatchResult> where TBatchPreResult : PreResultBatchBase, IDisposable, new()
    {
        protected readonly InferenceSession _session;
        protected readonly SessionOptions _options;
        protected readonly RunOptions _runOptions;

        protected readonly FixedBuffer _inputFixedBuffer;

        protected readonly OnnxModel _onnxModel;
        protected OrtValue _inputOrtValue;
        protected readonly Stopwatch _stopwatch;

        private readonly object _detectLock = new();
        protected MatBufferPool _matPool;
        protected readonly Mat _resizedImg;
        private int _batchPoolSize = 0;
        protected YoloConfig _config;

        protected readonly Lazy<ObjectPool<TBatchPreResult>> _preResultPool;
        protected readonly Lazy<ObjectPool<InferModel>> _inferModelPool;

        protected abstract List<TResult> RunBatchInfer(TBatchPreResult preResult);

        protected abstract InferModel RunInfer(TBatchPreResult preResult, long startTime);
        protected abstract TBatchResult PostprocessChannel(InferModel inferModel);

        protected abstract TBatchPreResult GetPreprocessImageBatchData(Mat inputImage, ImageBatchData imageBatchData, string imagePath);

        protected abstract TBatchResult PostprocessModel(TBatchPreResult preResult, long startTime);
        protected abstract Task RunBatchInfer(TBatchResult[] batchResults, int idx, TBatchPreResult item, long startTime, IBatchProcessCallback<TBatchResult> processCallback, Action<TBatchResult> receiveAction);


        public OnnxInferenceCore(InferenceSession session, SessionOptions options, OnnxModel onnxModel, YoloConfig config)
        {
            _resizedImg = new Mat();
            _config = config;
            _onnxModel = onnxModel;
            _stopwatch = new Stopwatch();
            _session = session;
            _options = options;
            _runOptions = new RunOptions();

            _inputFixedBuffer = new FixedBuffer(_onnxModel.InputShapeSize);

            _inputOrtValue = OrtValue.CreateTensorValueWithData(OrtMemoryInfo.DefaultInstance, TensorElementType.Float,
               _onnxModel.InputShape, _inputFixedBuffer.Address, _onnxModel.InputSizeInBytes);

            _preResultPool = new Lazy<ObjectPool<TBatchPreResult>>(() => new ObjectPool<TBatchPreResult>(() => new TBatchPreResult(), _config.BatchPoolSize));
            _inferModelPool = new Lazy<ObjectPool<InferModel>>(() => new ObjectPool<InferModel>(() => new InferModel(), _config.BatchPoolSize));
        }

      
        public void InitBufferPool(int batchPoolSize)
        {
            if (batchPoolSize != _batchPoolSize)
            {
                lock (_detectLock)
                {
                    if (batchPoolSize != _batchPoolSize)
                    {
                        _matPool?.Dispose();
                        _matPool = null;
                        _batchPoolSize = batchPoolSize;
                    }
                }
            }

            if (_matPool == null)
            {
                lock (_detectLock)
                {
                    if (_matPool == null)
                    {
                        _matPool = new MatBufferPool(batchPoolSize, _onnxModel);
                    }
                }
            }
        }

        public int BufferPoolUsedCount
        {
            get
            {
                if (_matPool == null)
                {
                    return 0;
                }
                return _matPool.UsedCount;
            }
        }

        protected IEnumerable<string[]> GetPreprocessWorkersSize(List<string> listImg)
        {
            int preprocessWorkers = Environment.ProcessorCount;
            if (_onnxModel.DeviceType == DeviceType.CPU)
            {
                preprocessWorkers = 2;
            }
            else
            {
                if (listImg.Count < Environment.ProcessorCount)
                {
                    preprocessWorkers = Environment.ProcessorCount / 2;
                }
                if (listImg.Count < preprocessWorkers)
                {
                    preprocessWorkers = 2;
                }
            }
            int size = listImg.Count / preprocessWorkers;

            if (size < 1)
            {
                size = listImg.Count;
            }
            return listImg.Chunk(size);
        }


        protected async Task PreprocessBatch(List<string> listImg, ChannelWriter<TBatchPreResult> writer)
        {
            var arr = GetPreprocessWorkersSize(listImg);
            Task[] tasks = new Task[arr.Count()];
            int idx = 0;
            foreach (string[] subList in arr)
            {
                tasks[idx++] = RunPreprocessSplitAsync(subList, writer);
            }
            await Task.WhenAll(tasks).ContinueWith(t =>
            {
                writer.Complete();
            });


        }
        private async Task RunPreprocessSplitAsync(IEnumerable<string> list, ChannelWriter<TBatchPreResult> writer)
        {
            await Task.Run(async () =>
            {
                foreach (string imgPath in list)
                {
                    var res = PreprocessImageChannel(imgPath);
                    await writer.WriteAsync(res);
                }

            });
        }
        public TBatchPreResult PreprocessImageChannel(string imagePath)
        {
            using Mat img = Cv2.ImRead(imagePath);
            return PreprocessImageChannel(img, imagePath);
        }

        public TBatchPreResult PreprocessImageChannel(Mat img, string imagePath)
        {
            var data = _matPool.Rent();
            return GetPreprocessImageBatchData(img, data, imagePath);
        }


        public async Task<TBatchResult[]> BatchRunAsync(List<string> listImg, IBatchProcessCallback<TBatchResult> processCallback, Action<TBatchResult> receiveAction)
        {
            var (producer, consumer, results) = BatchRunBaseFunc(listImg, processCallback, receiveAction);
            await Task.WhenAll(producer, consumer);
            return results;
        }
        public TBatchResult[] BatchRun(List<string> listImg, IBatchProcessCallback<TBatchResult> processCallback, Action<TBatchResult> receiveAction)
        {
            var (producer, consumer, results) = BatchRunBaseFunc(listImg, processCallback, receiveAction);
            Task.WaitAll(producer, consumer);
            return results;
        }

        public async Task<TBatchResult[]> BatchRunAsyncPostSync(List<string> listImg, IBatchProcessCallback<TBatchResult> processCallback, Action<TBatchResult> receiveAction)
        {
            var (producer, consumer, results) = BatchRunBasePostSync(listImg, processCallback, receiveAction);
            await Task.WhenAll(producer, consumer);
            return results;
        }
        public TBatchResult[] BatchRunPostSync(List<string> listImg, IBatchProcessCallback<TBatchResult> processCallback, Action<TBatchResult> receiveAction)
        {
            var (producer, consumer, results) = BatchRunBasePostSync(listImg, processCallback, receiveAction);
            Task.WaitAll(producer, consumer);
            return results;
        }
        private (Task producer, Task consumer, TBatchResult[] results) BatchRunBaseFunc(List<string> listImg, IBatchProcessCallback<TBatchResult> processCallback, Action<TBatchResult> receiveAction)
        {
            InitBufferPool(_config.BatchPoolSize);

            TBatchResult[] batchResults = new TBatchResult[listImg.Count];
            Channel<TBatchPreResult> channel = Channel.CreateBounded<TBatchPreResult>(YoloUtils.GetChannelOptions(_config.BatchPoolSize));

            var producer = PreprocessBatch(listImg, channel.Writer);

            var consumer = Task.Run(async () =>
            {
                ConcurrentBag<Task> tasks = new ConcurrentBag<Task>();
                int idx = 0;
                await foreach (TBatchPreResult item in channel.Reader.ReadAllAsync())
                {
                    long startTime = DateTimeOffset.Now.ToUnixTimeMilliseconds();
                    tasks.Add(RunBatchInfer(batchResults, idx++, item, startTime, processCallback, receiveAction));
                }
                await Task.WhenAll(tasks);
            });
            return (producer, consumer, batchResults);
        }

        private (Task producer, Task consumer, TBatchResult[] results) BatchRunBasePostSync(List<string> listImg, IBatchProcessCallback<TBatchResult> processCallback, Action<TBatchResult> receiveAction)
        {
            InitBufferPool(_config.BatchPoolSize);

            TBatchResult[] batchResults = new TBatchResult[listImg.Count];
            Channel<TBatchPreResult> channel = Channel.CreateBounded<TBatchPreResult>(YoloUtils.GetChannelOptions(_config.BatchPoolSize));

            var producer = PreprocessBatch(listImg, channel.Writer);

            var consumer = Task.Run(async () =>
            {
                int idx = 0;
                await foreach (TBatchPreResult item in channel.Reader.ReadAllAsync())
                {
                    long startTime = DateTimeOffset.Now.ToUnixTimeMilliseconds();
                    var modelResult = PostprocessModel(item, startTime);
                    batchResults[idx++] = modelResult;

                    _ = InferCompleteAsync(modelResult, processCallback, receiveAction);
                }
            });
            return (producer, consumer, batchResults);
        }

        public async IAsyncEnumerable<TBatchResult> BatchRunForeachAsync(List<string> listImg)
        {
            InitBufferPool(_config.BatchPoolSize);

            Channel<TBatchPreResult> channel = Channel.CreateBounded<TBatchPreResult>(YoloUtils.GetChannelOptions(_config.BatchPoolSize));
            Channel<InferModel> postChannel = Channel.CreateBounded<InferModel>(YoloUtils.GetChannelOptions(_config.BatchPoolSize));

            _ = PreprocessBatch(listImg, channel.Writer);
            _ = ReaderForeach(channel, postChannel).ContinueWith(t =>
             {
                 postChannel.Writer.Complete();
             });

            await foreach (InferModel item in postChannel.Reader.ReadAllAsync())
            {
                yield return PostprocessChannel(item);
            }

        }

        public async IAsyncEnumerable<TBatchResult> BatchRunForeachSync(List<string> listImg)
        {
            InitBufferPool(_config.BatchPoolSize);

            Channel<TBatchPreResult> channel = Channel.CreateBounded<TBatchPreResult>(YoloUtils.GetChannelOptions(_config.BatchPoolSize));

            _ = PreprocessBatch(listImg, channel.Writer);
            await foreach (TBatchPreResult item in channel.Reader.ReadAllAsync())
            {
                long startTime = DateTimeOffset.Now.ToUnixTimeMilliseconds();
                var modelResult = PostprocessModel(item, startTime);
                yield return modelResult;
            }

        }

        private async Task ReaderForeach(Channel<TBatchPreResult> channel, Channel<InferModel> postChannel)
        {
            await foreach (TBatchPreResult item in channel.Reader.ReadAllAsync())
            {
                long startTime = DateTimeOffset.Now.ToUnixTimeMilliseconds();
                var result = RunInfer(item, startTime);
                await postChannel.Writer.WriteAsync(result);
            }
        }


        protected async Task InferCompleteAsync(TBatchResult result, IBatchProcessCallback<TBatchResult> processCallback, Action<TBatchResult> receiveAction)
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

        public void ReturnBatchPreResult(TBatchPreResult preResult)
        {
            _preResultPool.Value.Return(preResult);
        }

        public void DisposeCore()
        {
            _matPool?.Dispose();

            _inputFixedBuffer.Dispose();

            _runOptions.Dispose();
            _session.Dispose();
            _options.Dispose();

            _runOptions.Dispose();
            _inputOrtValue.Dispose();

            if (_preResultPool.IsValueCreated)
            {
                _preResultPool.Value.Dispose();
            }

            if (_inferModelPool.IsValueCreated)
            {
                _inferModelPool.Value.Dispose();
            }
        }
    }
}
