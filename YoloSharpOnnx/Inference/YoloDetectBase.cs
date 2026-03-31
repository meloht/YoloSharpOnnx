using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using OpenCvSharp.Dnn;
using System;
using System.Buffers;
using System.Collections.Generic;
using System.Diagnostics;
using System.Reflection.Emit;
using System.Text;
using System.Threading.Channels;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Models;
using static System.Net.Mime.MediaTypeNames;
using static System.Net.WebRequestMethods;

namespace YoloSharpOnnx.Inference
{
    public class YoloDetectBase
    {
        protected readonly InferenceSession _session;
        protected readonly SessionOptions _options;
        protected readonly RunOptions _runOptions;


        protected readonly FixedBuffer _inputFixedBuffer;
        protected readonly FixedBuffer _outputFixedBuffer;
        protected readonly IPostprocess _postprocess;
        protected readonly IPreprocess _preprocess;

        protected readonly OnnxModel _onnxModel;

        protected OrtValue _inputOrtValue;


        protected readonly Stopwatch _stopwatch;
        public event EventHandler<DetectionBatchResult> BatchDetectItemCompleted;

        private readonly object _detectLock = new();
        protected MatBufferPool _matPool;
        protected Mat _resizedImg;
        private int _batchPoolSize = 0;

        public YoloDetectBase(InferenceSession session, SessionOptions options, IPostprocess postprocess, IPreprocess preprocess, OnnxModel onnxModel)
        {
            _resizedImg = new Mat();
            _onnxModel = onnxModel;
            _stopwatch = new Stopwatch();
            this._session = session;
            this._options = options;
            _runOptions = new RunOptions();

            _inputFixedBuffer = new FixedBuffer((int)_onnxModel.InputShapeSize);
            _outputFixedBuffer = new FixedBuffer((int)_onnxModel.OutputShapeSize);

            _postprocess = postprocess;
            _preprocess = preprocess;

            _inputOrtValue = OrtValue.CreateTensorValueWithData(OrtMemoryInfo.DefaultInstance, TensorElementType.Float,
               _onnxModel.InputShape, _inputFixedBuffer.Address, _onnxModel.InputSizeInBytes);

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

        public void DisposeBase()
        {
            _resizedImg.Dispose();
            _matPool?.Dispose();

            _inputFixedBuffer.Dispose();
            _outputFixedBuffer.Dispose();
            _runOptions.Dispose();
            _session.Dispose();
            _options.Dispose();

            _runOptions.Dispose();
            _inputOrtValue.Dispose();
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
        protected async Task PreprocessBatch(List<string> listImg, InterpolationFlags interpolationFlags, ChannelWriter<PreResultBatch> writer)
        {

            int preprocessWorkers = Environment.ProcessorCount / 2;
            if (_onnxModel.DeviceType == DeviceType.CPU || preprocessWorkers < 1)
            {
                preprocessWorkers = 2;
            }

            int size = listImg.Count / preprocessWorkers;
            if (size < 3)
            {
                await RunPreprocessSplitAsync(listImg, interpolationFlags, writer);
            }
            else
            {
                var arr = listImg.Chunk(size);
                Task[] tasks = new Task[arr.Count()];
                int idx = 0;
                foreach (string[] subList in arr)
                {
                    tasks[idx++] = RunPreprocessSplitAsync(subList, interpolationFlags, writer);
                }
                await Task.WhenAll(tasks);
            }

            writer.Complete();
        }
        private async Task RunPreprocessSplitAsync(IEnumerable<string> list, InterpolationFlags interpolationFlags, ChannelWriter<PreResultBatch> writer)
        {
            await Task.Run(async () =>
            {
                foreach (string imgPath in list)
                {
                    var res = PreprocessImageChannel(imgPath, interpolationFlags);
                    await writer.WriteAsync(res);
                }

            });
        }
        public PreResultBatch PreprocessImageChannel(string imagePath, InterpolationFlags interpolationFlags)
        {
            using Mat img = Cv2.ImRead(imagePath);
            return PreprocessImageChannel(img, imagePath, interpolationFlags);
        }

        public PreResultBatch PreprocessImageChannel(Mat img, string imagePath, InterpolationFlags interpolationFlags)
        {
            var data = _matPool.Rent();
            var res = _preprocess.PreprocessImage(img, data.ResizedImg, data.FixedBuffer, interpolationFlags);
            return new PreResultBatch(res, imagePath, data);
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
            if (_onnxModel.DeviceType == DeviceType.CPU)
            {
                channelOptions.SingleReader = true;
            }

            return channelOptions;
        }
        protected async Task<DetectionBatchResult[]> BatchDetectBaseAsync(List<string> listImg, IBatchProcessCallback processCallback, Action<DetectionBatchResult> receiveAction, YoloConfig yoloConfig, IBatchDetect batchDetect)
        {
            InitBufferPool(yoloConfig.BatchPoolSize);
            int idx = 0;
            DetectionBatchResult[] batchResults = new DetectionBatchResult[listImg.Count];
            var ChannelOptions = GetChannelOptions(yoloConfig.BatchPoolSize);
            Channel<PreResultBatch> channel = Channel.CreateBounded<PreResultBatch>(ChannelOptions);
            // Producer/consumer
            ChannelWriter<PreResultBatch> writer = channel.Writer;
            ChannelReader<PreResultBatch> reader = channel.Reader;

            var producer = PreprocessBatch(listImg, yoloConfig.ResizeAlgorithm, writer);

            var consumer = Task.Run(async () =>
            {

                await foreach (PreResultBatch item in reader.ReadAllAsync())
                {
                    long startTime = DateTimeOffset.Now.ToUnixTimeMilliseconds();
                    var result = batchDetect.RunBatchDetect(item, yoloConfig);
                    var modelResult = new DetectionBatchResult(item.ImagePath, result, startTime);
                    batchResults[idx] = modelResult;
                    Interlocked.Increment(ref idx);
                    _ = InferCompleteAsync(modelResult, processCallback, receiveAction);
                }
            });
            await Task.WhenAll(producer, consumer);
            return batchResults;
        }

        private async Task InferCompleteAsync(DetectionBatchResult result, IBatchProcessCallback processCallback, Action<DetectionBatchResult> receiveAction)
        {
            if (BatchDetectItemCompleted != null)
            {
                await Task.Run(async () =>
                {
                    BatchDetectItemCompleted(this, result);
                });
            }

            if (processCallback != null)
            {
                await Task.Run(async () =>
                 {
                     processCallback.ReceiveProcessResult(result);
                 });
            }
            if (receiveAction != null)
            {
                await Task.Run(async () =>
                {
                    receiveAction(result);
                });
            }
        }


        public void DrawDetections(Mat inputImage, List<DetectionResult> list)
        {
            foreach (var item in list)
            {
                DrawDetections(inputImage, item.Box, item.Confidence, item.ClassId, item.ClassName);
            }
        }
        public void DrawDetections(Mat img, Rect box, float score, int classId, string className)
        {
            var color = _onnxModel.ColorPalette[classId];

            double fontScale = 1.0;
            // 绘制边界框
            Cv2.Rectangle(img, box, color, 2);

            int height = img.Height;
            int width = img.Width;

            // 绘制标签
            string label = $"{className}: {score:F2}";
            int fontThick = 2;
            var textSize = Cv2.GetTextSize(label, HersheyFonts.HersheySimplex, fontScale, fontThick, out int baseline);
            var labelTop = new OpenCvSharp.Point(box.X, box.Y - 10);

            if (labelTop.Y < textSize.Height)
                labelTop.Y = box.Y + 10;

            if (labelTop.X + textSize.Width > width)
            {
                labelTop.X = labelTop.X - (labelTop.X + textSize.Width - width) - 4;
            }

            // 标签背景
            Cv2.Rectangle(img,
                new OpenCvSharp.Point(labelTop.X - 1, labelTop.Y - 8 - textSize.Height),
                new OpenCvSharp.Point(labelTop.X + textSize.Width, labelTop.Y + baseline),
                color, -1);

            // 标签文本
            Cv2.PutText(img, label, labelTop, HersheyFonts.HersheySimplex, fontScale, Scalar.White, fontThick, LineTypes.AntiAlias);
        }
    }
}
