using OpenCvSharp;
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
    public class SegPostprocessBase
    {
        protected readonly LabelModel[] _labels;
        protected readonly YoloConfig _yoloConfig;
        protected const float _threshold = 0.5f;
        protected int _inputSizeW;
        protected int _inputSizeH;
        protected readonly int _protoH;
        protected readonly int _protoW;
        protected readonly int _maskDim;

        public SegPostprocessBase(OnnxModel onnx, YoloConfig yoloConfig)
        {
            _labels = onnx.Labels;
            _yoloConfig = yoloConfig;
            _inputSizeH = onnx.InputHeight;
            _inputSizeW = onnx.InputWidth;
            _protoH = (int)onnx.OutputShape1[2];// [1,32,160,160] 160
            _protoW = (int)onnx.OutputShape1[3];//[1,32,160,160] 160
            _maskDim = (int)onnx.OutputShape1[1];//[1,32,160,160]  32 
        }


        protected SegResult BuildResult(Rect box, int classId, float score, ReadOnlySpan<float> maskCoeffs, ReadOnlySpan<float> output1, PreDetectResult preResult)
        {
            using Mat protoMask = new Mat(_protoH, _protoW, MatType.CV_32FC1);

            // STEP1：mask = coeff @ proto
            // 矩阵乘法：maskCoeffs(32) · protos(32, 160*160) → 160*160
            DecodeMask(protoMask, maskCoeffs, output1);

            // ====================== 4. 掩码缩放 + 二值化 ======================
            var maskRes = ScaleMaskToOriginal(protoMask, preResult, box);

            return new SegResult
            {
                Box = box,
                Confidence = score,
                ClassId = classId,
                ClassName = _labels[classId].Name,
                Mask = maskRes
            };
        }

        protected unsafe void DecodeMask(Mat protoMask, ReadOnlySpan<float> maskCoeffs, ReadOnlySpan<float> output1)
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
        protected Mat ScaleMaskToOriginal(Mat mask, PreDetectResult preResult, Rect box)
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
        protected static float Sigmoid(float x)
        {
            return 1.0f / (1.0f + (float)Math.Exp(-x));
        }
    }
}
