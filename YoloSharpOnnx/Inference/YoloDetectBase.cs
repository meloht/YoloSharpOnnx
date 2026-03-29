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

        protected readonly Scalar _paddingColor;


        protected readonly FixedBuffer _inputFixedBuffer;
        protected readonly FixedBuffer _outputFixedBuffer;
        protected readonly IPostprocess _postprocess;
        protected readonly OnnxModel _onnxModel;

        protected OrtValue _inputOrtValue;


        protected readonly Stopwatch _stopwatch;
        public event EventHandler<DetectionBatchResult> BatchDetectItemCompleted;

        private readonly object _detectLock = new();
        protected MatBufferPool _matPool;
        protected Mat _resizedImg;

        public YoloDetectBase(InferenceSession session, SessionOptions options, IPostprocess postprocess, OnnxModel onnxModel)
        {
            _resizedImg = new Mat();
            _onnxModel = onnxModel;
            _stopwatch = new Stopwatch();
            this._session = session;
            this._options = options;
            _runOptions = new RunOptions();

            _paddingColor = new Scalar(114, 114, 114);

            _inputFixedBuffer = new FixedBuffer((int)_onnxModel.InputShapeSize);
            _outputFixedBuffer = new FixedBuffer((int)_onnxModel.OutputShapeSize);

            _postprocess = postprocess;

            _inputOrtValue = OrtValue.CreateTensorValueWithData(OrtMemoryInfo.DefaultInstance, TensorElementType.Float,
               _onnxModel.InputShape, _inputFixedBuffer.Address, _onnxModel.InputSizeInBytes);

        }

        private void InitBufferPool(int batchPoolSize)
        {
            lock (_detectLock)
            {
                if (_matPool == null)
                {
                    _matPool = new MatBufferPool(batchPoolSize, _onnxModel);
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


        protected PreResult PreprocessImage(Mat inputImage, Mat resizedImg, FixedBuffer buffer, InterpolationFlags interpolationFlags)
        {

            // 1. 获取原始图像尺寸
            int imgH = inputImage.Height;
            int imgW = inputImage.Width;

            // 2. 计算缩放比例（按最小比例缩放，避免图像畸变）
            float scale = Math.Min((float)_onnxModel.InputHeight / imgH, (float)_onnxModel.InputWidth / imgW);

            // 3. 计算缩放后的尺寸（确保按比例缩放）
            int newImgW = (int)Math.Round(imgW * scale);
            int newImgH = (int)Math.Round(imgH * scale);

            // 4. 计算填充值（左右填充、上下填充，确保最终尺寸=1280×1280）
            int padW = (_onnxModel.InputWidth - newImgW) / 2; // 左右填充的一半
            int padH = (_onnxModel.InputHeight - newImgH) / 2; // 上下填充的一半


            // 5. 缩放图像（若原始尺寸≠缩放后尺寸）

            Cv2.Resize(inputImage, resizedImg, new OpenCvSharp.Size(newImgW, newImgH), interpolation: interpolationFlags);

            // BGR转RGB
            Cv2.CvtColor(resizedImg, resizedImg, ColorConversionCodes.BGR2RGB);

            Cv2.CopyMakeBorder(
               src: resizedImg,
               dst: resizedImg,
               top: padH,        // 顶部填充
               bottom: _onnxModel.InputHeight - newImgH - padH, // 底部填充（补全到 1280）
               left: padW,       // 左侧填充
               right: _onnxModel.InputWidth - newImgW - padW,  // 右侧填充（补全到 1280）
               borderType: BorderTypes.Constant,
               value: _paddingColor // 填充色（BGR 格式）
           );

            GetChwArrPointer(resizedImg, buffer);

            // 添加批次维度 (1, 3, H, W)
            return new PreResult(imgH, imgW, padH, padW, scale);
        }
        public void GetChwArrPointer(Mat paddedImg, FixedBuffer buffer)
        {
            int height = paddedImg.Height;
            int width = paddedImg.Width;
            int channels = paddedImg.Channels();

            unsafe
            {
                int index = 0;
                byte* ptr = (byte*)paddedImg.DataPointer;
                float* data = buffer.Pointer;
                for (int c = 0; c < channels; c++)
                {
                    for (int y = 0; y < height; y++)
                    {
                        for (int x = 0; x < width; x++)
                        {
                            data[index++] = ptr[(y * width + x) * channels + c] / 255.0f;
                        }

                    }
                }
            }
        }


        protected async Task PreprocessBatch(List<string> listImg, InterpolationFlags interpolationFlags, ChannelWriter<PreResultBatch> writer)
        {

            int preprocessWorkers = Environment.ProcessorCount / 2;
            int size = listImg.Count / preprocessWorkers;
            if (size < 3)
            {
                await RunPreprocessSplitAsync(listImg, interpolationFlags, writer);
            }
            else
            {
                var arr = listImg.Chunk(size);
                foreach (string[] subList in arr)
                {
                    await RunPreprocessSplitAsync(subList, interpolationFlags, writer);
                }
            }

            writer.Complete();
        }
        private async Task RunPreprocessSplitAsync(IEnumerable<string> list, InterpolationFlags interpolationFlags, ChannelWriter<PreResultBatch> writer)
        {

            await Task.Run(async () =>
            {
                foreach (string imgPath in list)
                {
                    var data = _matPool.Rent();
                    using Mat img = Cv2.ImRead(imgPath);
                    var res = PreprocessImage(img, data.ResizedImg, data.FixedBuffer, interpolationFlags);
                    await writer.WriteAsync(new PreResultBatch(res, imgPath, data));
                }

            });
        }

        protected async Task<DetectionBatchResult[]> BatchDetectBaseAsync(List<string> listImg, IBatchProcessCallback processCallback, Action<DetectionBatchResult> receiveAction, int batchPoolSize, YoloConfiguration yoloConfig, IBatchDetect batchDetect)
        {
            InitBufferPool(batchPoolSize);
            int idx = 0;
            DetectionBatchResult[] batchResults = new DetectionBatchResult[listImg.Count];

            Channel<PreResultBatch> channel = Channel.CreateBounded<PreResultBatch>(batchPoolSize);
            // Producer/consumer
            ChannelWriter<PreResultBatch> writer = channel.Writer;
            ChannelReader<PreResultBatch> reader = channel.Reader;

            var producer = PreprocessBatch(listImg, yoloConfig.ResizeAlgorithm, writer);

            var consumer = Task.Run(async () =>
            {

                await foreach (PreResultBatch item in reader.ReadAllAsync())
                {
                    var result = batchDetect.RunBatchDetect(item, yoloConfig);
                    var modelResult = new DetectionBatchResult(item.ImagePath, result);
                    batchResults[idx] = modelResult;
                    Interlocked.Increment(ref idx);
                    await InferCompleteAsync(modelResult, processCallback, receiveAction);


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

        protected LabelModel[] GetModelLabels(InferenceSession session)
        {
            var metaData = session.ModelMetadata.CustomMetadataMap;
            var onnxLabelData = metaData["names"];
            // Labels to Dictionary
            var onnxLabels = onnxLabelData
                .Trim('{', '}')
                .Replace("'", "")
                .Split(", ")
                .Select(x => x.Split(": "))
                .ToDictionary(x => int.Parse(x[0]), x => x[1]);

            return [.. onnxLabels!.Select((label, index) => new LabelModel
            {
                Index = index,
                Name = label.Value,
            })];
        }
        protected Scalar[] GenerateColorPalette(int count)
        {
            var rng = new Random();
            var palette = new Scalar[count];
            var colors = ColorTemplate.Get();
            for (int i = 0; i < count; i++)
            {
                palette[i] = ColorTemplate.HexToRgbaScalar(colors[i % count]);
            }
            return palette;
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
                new OpenCvSharp.Point(labelTop.X - 8, labelTop.Y - 8 - textSize.Height),
                new OpenCvSharp.Point(labelTop.X + textSize.Width, labelTop.Y + baseline),
                color, -1);

            // 标签文本
            Cv2.PutText(img, label, labelTop, HersheyFonts.HersheySimplex, fontScale, Scalar.White, fontThick, LineTypes.AntiAlias);
        }
    }
}
