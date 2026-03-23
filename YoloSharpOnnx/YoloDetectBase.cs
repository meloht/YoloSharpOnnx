using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Reflection.Emit;
using System.Text;
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


        public YoloDetectBase(InferenceSession session, SessionOptions options)
        {
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
