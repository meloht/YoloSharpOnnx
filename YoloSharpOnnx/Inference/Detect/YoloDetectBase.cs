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


namespace YoloSharpOnnx.Inference.Detect
{
    public class YoloDetectBase : OnnxInferenceCore
    {
        protected readonly IDetPostprocess _postprocess;
        protected readonly IDetPreprocess _preprocess;

        public event EventHandler<DetectionBatchResult> BatchDetectItemCompleted;


        public YoloDetectBase(InferenceSession session, SessionOptions options, IDetPostprocess postprocess, IDetPreprocess preprocess, OnnxModel onnxModel, YoloConfig config)
            : base(session, options, onnxModel, config)
        {
            _postprocess = postprocess;
            _preprocess = preprocess;

        }


        protected async Task PreprocessBatch(List<string> listImg, InterpolationFlags interpolationFlags, ChannelWriter<PreResultBatch> writer)
        {
            var arr = GetPreprocessWorkersSize(listImg);
            Task[] tasks = new Task[arr.Count()];
            int idx = 0;
            foreach (string[] subList in arr)
            {
                tasks[idx++] = RunPreprocessSplitAsync(subList, interpolationFlags, writer);
            }
            await Task.WhenAll(tasks).ContinueWith(t =>
            {
                writer.Complete();
            });


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

            return channelOptions;
        }
        protected async Task<DetectionBatchResult[]> BatchDetectBaseAsync(List<string> listImg, IBatchProcessCallback processCallback, Action<DetectionBatchResult> receiveAction, IBatchDetect batchDetect)
        {
            InitBufferPool(_config.BatchPoolSize);
            int idx = 0;
            DetectionBatchResult[] batchResults = new DetectionBatchResult[listImg.Count];
            var ChannelOptions = GetChannelOptions(_config.BatchPoolSize);
            Channel<PreResultBatch> channel = Channel.CreateBounded<PreResultBatch>(ChannelOptions);

            var producer = PreprocessBatch(listImg, _config.ResizeAlgorithm, channel.Writer);

            var consumer = Task.Run(async () =>
            {
                await foreach (PreResultBatch item in channel.Reader.ReadAllAsync())
                {
                    long startTime = DateTimeOffset.Now.ToUnixTimeMilliseconds();
                    var result = batchDetect.RunBatchDetect(item);
                    var modelResult = new DetectionBatchResult(item.ImagePath, result, startTime);
                    batchResults[idx++] = modelResult;
                    //Interlocked.Increment(ref idx);
                    _ = InferCompleteAsync(modelResult, processCallback, receiveAction);
                }
            });
            await Task.WhenAll(producer, consumer);
            return batchResults;
        }

        protected async IAsyncEnumerable<DetectionBatchResult> BatchDetectBaseForeachAsync(List<string> listImg, IBatchDetect batchDetect)
        {
            InitBufferPool(_config.BatchPoolSize);

            var ChannelOptions = GetChannelOptions(_config.BatchPoolSize);
            Channel<PreResultBatch> channel = Channel.CreateBounded<PreResultBatch>(ChannelOptions);

            _ = PreprocessBatch(listImg, _config.ResizeAlgorithm, channel.Writer);
            await foreach (PreResultBatch item in channel.Reader.ReadAllAsync())
            {
                long startTime = DateTimeOffset.Now.ToUnixTimeMilliseconds();
                var result = batchDetect.RunBatchDetect(item);
                var modelResult = new DetectionBatchResult(item.ImagePath, result, startTime);
                yield return modelResult;
            }

        }

        private async Task InferCompleteAsync(DetectionBatchResult result, IBatchProcessCallback processCallback, Action<DetectionBatchResult> receiveAction)
        {
            if (BatchDetectItemCompleted != null)
            {
                await Task.Run(() =>
                {
                    BatchDetectItemCompleted(this, result);
                });
            }

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

            int x = box.X;
            int y = box.Y - 10; ;
            if (y < textSize.Height)
                y = box.Y + 10;

            if (x + textSize.Width > width)
            {
                x = x - (x + textSize.Width - width) - 4;
            }

            // 标签背景
            Cv2.Rectangle(img,
                new OpenCvSharp.Point(x - 1, y - 8 - textSize.Height),
                new OpenCvSharp.Point(x + textSize.Width, y + baseline),
                color, -1);

            // 标签文本
            Cv2.PutText(img, label, new Point(x + 1, y), HersheyFonts.HersheySimplex, fontScale, Scalar.White, fontThick, LineTypes.AntiAlias);
        }
    }
}
