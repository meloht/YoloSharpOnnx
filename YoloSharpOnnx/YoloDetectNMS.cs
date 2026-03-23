using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;
using OpenCvSharp.Dnn;
using System;
using System.Collections.Generic;
using System.Reflection.Emit;
using System.Text;
using YoloSharpOnnx.Models;

namespace YoloSharpOnnx
{
    public class YoloDetectNMS : YoloDetectBase, IYoloDetect
    {

        private readonly int _boxNums;
        private readonly int _boxNums2;
        private readonly int _boxNums3;
        private readonly int _boxNums4;

        private List<Rect> _boxes = new List<Rect>();
        private List<float> _scores = new List<float>();
        private List<int> _classIds = new List<int>();

        public YoloDetectNMS(InferenceSession session, SessionOptions options)
            : base(session, options)
        {

            _boxNums = (int)_outputShape[2];
            _boxNums2 = _boxNums * 2;
            _boxNums3 = _boxNums * 3;
            _boxNums4 = _boxNums * 4;
        }

        protected PreResult Preprocess(Mat inputImage, float[] data, int inputHeight, int inputWidth, InterpolationFlags interpolationFlags)
        {
            // BGR转RGB
            using Mat rgbImg = new Mat();

            Cv2.CvtColor(inputImage, rgbImg, ColorConversionCodes.BGR2RGB);
            // 1. 获取原始图像尺寸
            int imgH = rgbImg.Rows;
            int imgW = rgbImg.Cols;

            // 2. 计算缩放比例（按最小比例缩放，避免图像畸变）
            float scale = Math.Min((float)inputHeight / imgH, (float)inputWidth / imgW);

            // 3. 计算缩放后的尺寸（确保按比例缩放）
            int newImgW = (int)Math.Round(imgW * scale);
            int newImgH = (int)Math.Round(imgH * scale);

            // 4. 计算填充值（左右填充、上下填充，确保最终尺寸=1280×1280）
            int padW = (inputWidth - newImgW) / 2; // 左右填充的一半
            int padH = (inputHeight - newImgH) / 2; // 上下填充的一半

            // 5. 缩放图像（若原始尺寸≠缩放后尺寸）
            using Mat resizedImg = rgbImg;
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
                bottom: inputHeight - newImgH - padH, // 底部填充（补全到 1280）
                left: padW,       // 左侧填充
                right: inputWidth - newImgW - padW,  // 右侧填充（补全到 1280）
                borderType: BorderTypes.Constant,
                value: _paddingColor // 填充色（BGR 格式）
            );

            // 关键检查：确保填充后尺寸严格为 1280×1280
            if (letterboxImg.Rows != inputHeight || letterboxImg.Cols != inputWidth)
            {
                throw new Exception($"Letterbox size error! expected (1280,1280)，actual ({letterboxImg.Rows},{letterboxImg.Cols})");
            }

            GetChwArr(letterboxImg, data);

            // 添加批次维度 (1, 3, H, W)
            return new PreResult(data, padH, padW);
        }




        private List<DetectionResult> Postprocess(int imageHeight, int imageWidth, ReadOnlySpan<float> ortSpan, int padTop, int padLeft, YoloConfiguration yoloConfig)
        {
            _boxes.Clear();
            _scores.Clear();
            _classIds.Clear();

            float gain = Math.Min((float)_inputHeight / imageHeight, (float)_inputWidth / imageWidth);
            for (int i = 0; i < _boxNums; i++)
            {
                // Move forward to confidence value of first label
                var labelOffset = i + _boxNums4;

                float bestConfidence = 0f;
                int bestLabelIndex = -1;

                // Get confidence and label for current bounding box
                for (var l = 0; l < _labels.Length; l++, labelOffset += _boxNums)
                {
                    var boxConfidence = ortSpan[labelOffset];

                    if (boxConfidence > bestConfidence)
                    {
                        bestConfidence = boxConfidence;
                        bestLabelIndex = l;
                    }
                }

                // Stop early if confidence is low
                if (bestConfidence < yoloConfig.Confidence)
                    continue;

                float x = ortSpan[i] - padLeft;
                float y = ortSpan[i + _boxNums] - padTop;
                float w = ortSpan[i + _boxNums2];
                float h = ortSpan[i + _boxNums3];

                // Calculate the scaled coordinates of the bounding box
                int left = (int)((x - w / 2) / gain);
                int top = (int)((y - h / 2) / gain);
                int width = (int)(w / gain);
                int height = (int)(h / gain);

                // Ensure coordinates are within image bounds
                left = Math.Max(0, left);
                top = Math.Max(0, top);
                width = Math.Min(width, imageWidth - left);
                height = Math.Min(height, imageHeight - top);

                // Add the class ID, score, and box coordinates to the respective lists
                if (width > 0 && height > 0)
                {
                    _classIds.Add(bestLabelIndex);
                    _scores.Add(bestConfidence);
                    _boxes.Add(new Rect(left, top, width, height));
                }
            }

            // 非极大值抑制
            int[] indices = [];
            if (_boxes.Count > 0)
            {
                CvDnn.NMSBoxes(_boxes, _scores, yoloConfig.Confidence, yoloConfig.IoU, out indices);
            }
            List<DetectionResult> results = new List<DetectionResult>();
            // 绘制检测结果
            foreach (var idx in indices)
            {
                Rect box = _boxes[idx];
                float score = _scores[idx];
                int class_id = _classIds[idx];
                string lable = _labels[class_id].Name;

                DetectionResult detection = new DetectionResult();
                detection.Confidence = score;
                detection.ClassName = lable;
                detection.ClassId = class_id;
                detection.Box = box;
                results.Add(detection);

            }

            return results;
        }

        public void Dispose()
        {
            _session.Dispose();
            _options.Dispose();
            _runOptions.Dispose();
        }

        public List<DetectionResult> Run(Mat inputImage, YoloConfiguration yoloConfig)
        {
            // 预处理图像
            var imgData = Preprocess(inputImage, _inputBuffer, _inputHeight, _inputWidth, yoloConfig.ResizeAlgorithm);

            using var inputOrtValue = OrtValue.CreateTensorValueFromMemory(imgData.OutData, _inputShape);

            using var runOptions = new RunOptions();
            // 执行推理
            using var outputs = _session.Run(runOptions, _session.InputNames, [inputOrtValue], _session.OutputNames);
            using var output0 = outputs[0];

            // 后处理
            var result = Postprocess(inputImage.Height, inputImage.Width, output0.GetTensorDataAsSpan<float>(), imgData.TopPad, imgData.LeftPad, yoloConfig);


            return result;
        }
    }
}
