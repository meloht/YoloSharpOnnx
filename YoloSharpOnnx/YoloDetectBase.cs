using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Reflection.Emit;
using System.Text;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Models;

namespace YoloSharpOnnx
{
    public class YoloDetectBase
    {
        protected readonly InferenceSession _session;
        protected readonly SessionOptions _options;
        protected readonly RunOptions _runOptions;

        protected readonly Scalar[] _colorPalette;
        protected readonly Scalar _paddingColor;
        protected readonly int _inputWidth;
        protected readonly int _inputHeight;

        protected readonly string _inputName;
        protected readonly string _outputName;

        protected readonly float[] _inputBuffer;
        protected readonly LabelModel[] _labels;

        protected readonly long[] _inputShape;
        protected readonly long[] _outputShape;

        protected readonly long _inputShapeSize;

        protected readonly Stopwatch _stopwatch;

        public YoloDetectBase(InferenceSession session, SessionOptions options)
        {
            _stopwatch = new Stopwatch();
            this._session = session;
            this._options = options;
            _runOptions = new RunOptions();

            _inputName = _session.InputNames[0];
            _outputName = _session.OutputNames[0];

            _paddingColor = new Scalar(114, 114, 114);

            var inputMeta = _session.InputMetadata;
            var outputMeta = _session.OutputMetadata;



            _inputShape = Array.ConvertAll<int, long>(inputMeta[_inputName].Dimensions, Convert.ToInt64);
            _outputShape = Array.ConvertAll<int, long>(outputMeta[_outputName].Dimensions, Convert.ToInt64);

            _inputHeight = (int)_inputShape[2];
            _inputWidth = (int)_inputShape[3];

            _inputShapeSize = ShapeUtils.GetSizeForShape(_inputShape);

            _inputBuffer = new float[_inputShapeSize];

            _labels = GetModelLabels(session);
            _colorPalette = GenerateColorPalette(_labels.Length);

        }


        protected PreResult Preprocess(Mat inputImage, float[] data, InterpolationFlags interpolationFlags)
        {
            // BGR转RGB
            using Mat rgbImg = new Mat();

            Cv2.CvtColor(inputImage, rgbImg, ColorConversionCodes.BGR2RGB);
            // 1. 获取原始图像尺寸
            int imgH = inputImage.Height;
            int imgW = inputImage.Width;

            // 2. 计算缩放比例（按最小比例缩放，避免图像畸变）
            float scale = Math.Min((float)_inputHeight / imgH, (float)_inputWidth / imgW);

            // 3. 计算缩放后的尺寸（确保按比例缩放）
            int newImgW = (int)Math.Round(imgW * scale);
            int newImgH = (int)Math.Round(imgH * scale);

            // 4. 计算填充值（左右填充、上下填充，确保最终尺寸=1280×1280）
            int padW = (_inputWidth - newImgW) / 2; // 左右填充的一半
            int padH = (_inputHeight - newImgH) / 2; // 上下填充的一半

            // 5. 缩放图像（若原始尺寸≠缩放后尺寸）
            using Mat resizedImg = new Mat();
            if (imgW != newImgW || imgH != newImgH)
            {
                Cv2.Resize(rgbImg, resizedImg, new OpenCvSharp.Size(newImgW, newImgH), interpolation: interpolationFlags);
            }

            // 6. 填充到 1280×1280（用 114 填充，YOLO 常用默认值）
            using Mat letterboxImg = new Mat();
            Cv2.CopyMakeBorder(
                src: resizedImg,
                dst: letterboxImg,
                top: padH,        // 顶部填充
                bottom: _inputHeight - newImgH - padH, // 底部填充（补全到 1280）
                left: padW,       // 左侧填充
                right: _inputWidth - newImgW - padW,  // 右侧填充（补全到 1280）
                borderType: BorderTypes.Constant,
                value: _paddingColor // 填充色（BGR 格式）
            );

            // 关键检查：确保填充后尺寸严格为 1280×1280
            if (letterboxImg.Rows != _inputHeight || letterboxImg.Cols != _inputWidth)
            {
                throw new Exception($"Letterbox size error! expected (1280,1280)，actual ({letterboxImg.Rows},{letterboxImg.Cols})");
            }

            GetChwArr(letterboxImg, data);

            // 添加批次维度 (1, 3, H, W)
            return new PreResult(data, padH, padW, scale);
        }
        public void GetChwArr(Mat paddedImg, float[] data)
        {
            int height = paddedImg.Height;
            int width = paddedImg.Width;
            int channels = paddedImg.Channels();
            int index = 0;
            for (int c = 0; c < channels; c++)          // 通道（R=0, G=1, B=2）
            {
                for (int h = 0; h < height; h++)  // 高度
                {
                    for (int w = 0; w < width; w++)  // 宽度
                    {
                        var vec = paddedImg.At<Vec3b>(h, w);
                        data[index++] = (vec[c] / 255.0f);
                    }
                }
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
            var color = _colorPalette[classId];

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
