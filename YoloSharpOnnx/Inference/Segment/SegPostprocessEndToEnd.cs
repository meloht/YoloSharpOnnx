using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;
using OpenCvSharp.Dnn;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Inference.Detect.Models;
using YoloSharpOnnx.Models;

namespace YoloSharpOnnx.Inference.Segment
{
    public class SegPostprocessEndToEnd : ISegPostprocess
    {
        private readonly LabelModel[] _labels;
        private readonly YoloConfig _yoloConfig;
        private const float _threshold = 0.5f;
        private int _inputSizeW;
        private int _inputSizeH;

        public SegPostprocessEndToEnd(OnnxModel onnx, YoloConfig yoloConfig)
        {
            _labels = onnx.Labels;
            _yoloConfig = yoloConfig;
            _inputSizeH = onnx.InputHeight;
            _inputSizeW = onnx.InputWidth;
        }

        public List<SegResult> PostProcess(OrtValue outputValue0, OrtValue outputValue1, PreDetectResult preResult)
        {
            List<SegResult> results = new List<SegResult>();

            int imgW = preResult.ImageWidth;
            int imgH = preResult.ImageHeight;

            var shape0 = outputValue0.GetTensorTypeAndShape().Shape; //  [1,300,38]  
            var shape1 = outputValue1.GetTensorTypeAndShape().Shape; //[1,32,160,160]

            var output0 = outputValue0.GetTensorDataAsSpan<float>();
            var output1 = outputValue1.GetTensorDataAsSpan<float>();

            int maskCoeff = (int)shape1[1];//32
            int maxDet = (int)shape0[1]; // 300

            int rowOffset = (int)shape0[2];// boxAttrs+maskCoeff

            int boxAttrs = (int)(shape0[2] - maskCoeff); //38-32=6


            int protoH = (int)shape1[2];//160
            int protoW = (int)shape1[3];//160


            // ====================== 1. 解析 output0 [1,300,38] ======================
            for (int i = 0; i < maxDet; i++)
            {
                // 定位当前目标在数组中的起始位置
                int offset = i * rowOffset;

                float score = output0[offset + 4];

                // 置信度过滤
                if (score < _yoloConfig.Confidence) continue;

                // 读取6个基础属性
                float x1 = (output0[offset + 0] - preResult.PadX) / preResult.Scale;
                float y1 = (output0[offset + 1] - preResult.PadY) / preResult.Scale;
                float x2 = (output0[offset + 2] - preResult.PadX) / preResult.Scale;
                float y2 = (output0[offset + 3] - preResult.PadY) / preResult.Scale;
                int classId = (int)output0[offset + 5];

                // 坐标裁剪到图像范围内
                Rect box = new Rect((int)x1, (int)y1, (int)(x2 - x1), (int)(y2 - y1));

                // ====================== 2. 读取 32 个掩码系数 ======================
                var maskCoeffs = output0.Slice(offset + boxAttrs, maskCoeff);//maskCoeffs(32)
                // ====================== 3. 解析 output1 原型掩码 [1,32,160,160] ======================

                using Mat protoMask = new Mat(protoH, protoW, MatType.CV_32FC1);

                // STEP1：mask = coeff @ proto
                // 矩阵乘法：maskCoeffs(32) · protos(32, 160*160) → 160*160
                DecodeMask(protoMask, maskCoeffs, output1);

                // ====================== 4. 掩码缩放 + 二值化 ======================
                var maskRes = ScaleMaskToOriginal(protoMask, preResult, box);

                results.Add(new SegResult
                {
                    Box = box,
                    Confidence = score,
                    ClassId = classId,
                    ClassName = _labels[classId].Name,
                    Mask = maskRes
                });
            }

            return results;

        }
        private unsafe void DecodeMask(Mat protoMask, ReadOnlySpan<float> maskCoeffs, ReadOnlySpan<float> output1)
        {
            int protoH = protoMask.Height;
            int protoW = protoMask.Width;

            float* ptr = (float*)protoMask.DataPointer;
            for (int y = 0; y < protoH; y++)
            {
                for (int x = 0; x < protoW; x++)
                {
                    float sum = 0;
                    for (int c = 0; c < maskCoeffs.Length; c++)
                    {
                        // output1 布局: [c, y, x]
                        sum += maskCoeffs[c] * output1[c * protoH * protoW + y * protoW + x];
                    }
                    ptr[y * protoW + x] = Sigmoid(sum); // sigmoid激活
                }
            }
        }
        private Mat ScaleMaskToOriginal(Mat mask, PreDetectResult preResult, Rect box)
        {
            // STEP2：resize 到模型输入尺寸
            using Mat upsampled = new Mat();
            Cv2.Resize(mask, upsampled, new OpenCvSharp.Size(_inputSizeW, _inputSizeH), interpolation: InterpolationFlags.Linear);

            // STEP3：去除 letterbox padding
            int left = (int)Math.Round(preResult.PadX - 0.1);
            int top = (int)Math.Round(preResult.PadY - 0.1);

            int validW = (int)Math.Round(preResult.ImageWidth * preResult.Scale);
            int validH = (int)Math.Round(preResult.ImageHeight * preResult.Scale);

            Rect roi = new Rect(
                left,
                top,
                Math.Min(validW, upsampled.Width - left),
                Math.Min(validH, upsampled.Height - top));

            using Mat noPad = new Mat(upsampled, roi);

            //-----------------------------------
            // STEP4：还原到原图尺寸
            //-----------------------------------
            using Mat restored = new Mat();
            Cv2.Resize(noPad, restored,
                new OpenCvSharp.Size(preResult.ImageWidth, preResult.ImageHeight), interpolation: InterpolationFlags.Linear);

            Rect safeBox = new Rect(
                 Math.Max(box.X, 0),
                 Math.Max(box.Y, 0),
                 Math.Min(box.Width, preResult.ImageWidth - box.X),
                 Math.Min(box.Height, preResult.ImageHeight - box.Y));


            using Mat binary = new Mat();
            Cv2.Threshold(restored, binary, _threshold, 255, ThresholdTypes.Binary);
            binary.ConvertTo(binary, MatType.CV_8UC1);

            Mat srcRoi = new Mat(binary, safeBox);
            return srcRoi;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float Sigmoid(float x)
        {
            return 1.0f / (1.0f + (float)Math.Exp(-x));
        }

    }
}
