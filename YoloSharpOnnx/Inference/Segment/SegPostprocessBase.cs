using OpenCvSharp;
using OpenCvSharp.Dnn;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Channels;
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
        private readonly int _hw;

        public SegPostprocessBase(OnnxModel onnx, YoloConfig yoloConfig)
        {
            _labels = onnx.Labels;
            _yoloConfig = yoloConfig;
            _inputSizeH = onnx.InputHeight;
            _inputSizeW = onnx.InputWidth;
            _protoH = (int)onnx.OutputShape1[2];// [1,32,160,160] 160
            _protoW = (int)onnx.OutputShape1[3];//[1,32,160,160] 160
            _maskDim = (int)onnx.OutputShape1[1];//[1,32,160,160]  32 

            _hw = _protoH * _protoW;
        }

        protected Mat GetMaskFromProto(ReadOnlySpan<float> output1)
        {
            using Mat protoMat = new Mat(_maskDim, _hw, MatType.CV_32FC1);
            unsafe
            {
                float* protoPtr = (float*)protoMat.DataPointer;

                for (int c = 0; c < _maskDim; c++)
                {
                    int srcOffset = c * _hw;
                    for (int i = 0; i < _hw; i++)
                    {
                        protoPtr[srcOffset + i] = output1[srcOffset + i];
                    }
                }
            }
            return protoMat;

        }
        protected Mat GetCoeffMat(ReadOnlySpan<float> maskCoeffs)
        {
            Mat coeffMat = new Mat(1, _maskDim, MatType.CV_32FC1);
            unsafe
            {
                float* coeffPtr = (float*)coeffMat.DataPointer;
                for (int i = 0; i < _maskDim; i++)
                {
                    coeffPtr[i] = maskCoeffs[i];
                }
            }
            return coeffMat;

        }

        protected void GEMM(List<SegResult> list, List<Mat> coeffMatList, ReadOnlySpan<float> output1, PreDetectResult preResult)
        {
            if (list.Count == 0)
                return;
            int count = list.Count;
            using Mat coeffMat = new Mat(count, _maskDim, MatType.CV_32FC1);
            unsafe
            {
                float* ptr = (float*)coeffMat.DataPointer;
                for (int i = 0; i < count; i++)
                {
                    float* coeff = (float*)coeffMatList[i].DataPointer;
                    int offset = i * _maskDim;
                    for (int c = 0; c < _maskDim; c++)
                    {
                        ptr[offset + c] = coeff[c];
                    }

                }
            }


            using Mat protoMat = new Mat(_maskDim, _hw, MatType.CV_32FC1);
            unsafe
            {
                float* ptr = (float*)protoMat.DataPointer;

                for (int i = 0; i < output1.Length; i++)
                {
                    ptr[i] = output1[i];
                }
            }

            using Mat masks = new Mat();

            Cv2.Gemm(coeffMat, protoMat, 1.0, InputArray.Create(new Mat()), 0.0, masks);

            Cv2.Multiply(masks, -1.0, masks);
            Cv2.Exp(masks, masks);
            Cv2.Add(masks, 1.0, masks);
            Cv2.Divide(1.0, masks, masks);
            Mat[] channels = new Mat[count];
            for (int i = 0; i < count; i++)
            {
                using Mat row = masks.Row(i);
                channels[i] = row.Reshape(1, _protoH).Clone();
                //list[i].Mask = ScaleMaskToOriginal(mask, preResult, list[i].Box);
            }
            using Mat merged = new Mat();
            Cv2.Merge(channels, merged);

            using Mat upsampled = new Mat();
            Cv2.Resize(merged, upsampled, new OpenCvSharp.Size(_inputSizeW, _inputSizeH), interpolation: InterpolationFlags.Linear);

            // 去除 letterbox padding
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
            //还原到原图尺寸
            //-----------------------------------
            using Mat restored = new Mat();
            Cv2.Resize(noPad, restored,
                new OpenCvSharp.Size(preResult.ImageWidth, preResult.ImageHeight), interpolation: InterpolationFlags.Linear);

            using Mat binary = new Mat();
            Cv2.Threshold(restored, binary, _threshold, 255, ThresholdTypes.Binary);

            using Mat mat8u = new Mat();
            binary.ConvertTo(mat8u, MatType.CV_8UC1);

            Mat[] finalChannels = Cv2.Split(mat8u);
            for (int i = 0; i < count; i++)
            {

                var box = list[i].Box;
                Rect safeBox = new Rect(
                  Math.Max(box.X, 0),
                  Math.Max(box.Y, 0),
                  Math.Min(box.Width, preResult.ImageWidth - Math.Abs(box.X)),
                  Math.Min(box.Height, preResult.ImageHeight - Math.Abs(box.Y)));

                list[i].Mask = new Mat(finalChannels[i], safeBox);
            }
        }


        protected unsafe void DecodeMask(Mat protoMask, ReadOnlySpan<float> maskCoeffs, ReadOnlySpan<float> output1)
        {
            //using Mat protoMask = new Mat(_protoH, _protoW, MatType.CV_32FC1);
            using Mat coeffMat = new Mat(1, _maskDim, MatType.CV_32FC1);

            float* coeffPtr = (float*)coeffMat.DataPointer;
            for (int i = 0; i < _maskDim; i++)
            {
                coeffPtr[i] = maskCoeffs[i];
            }
            using Mat protoMat = new Mat(_maskDim, _hw, MatType.CV_32FC1);

            float* protoPtr = (float*)protoMat.DataPointer;

            for (int c = 0; c < _maskDim; c++)
            {
                int srcOffset = c * _hw;
                for (int i = 0; i < _hw; i++)
                {
                    protoPtr[srcOffset + i] = output1[srcOffset + i];
                }
            }
            using Mat result = new Mat();

            Cv2.Gemm(coeffMat, protoMat, 1.0, InputArray.Create(new Mat()), 0.0, result);

            using Mat reshaped = result.Reshape(1, _protoH);

            float* src = (float*)reshaped.DataPointer;
            float* dst = (float*)protoMask.DataPointer;

            for (int i = 0; i < _hw; i++)
            {
                dst[i] = 1.0f / (1.0f + MathF.Exp(-src[i]));
            }


        }

        protected unsafe void DecodeMaskOrginal(Mat protoMask, ReadOnlySpan<float> maskCoeffs, ReadOnlySpan<float> output1)
        {
            float* ptr = (float*)protoMask.DataPointer;

            for (int c = 0; c < _maskDim; c++)
            {
                float coeff = maskCoeffs[c];
                int offset = c * _hw;

                for (int i = 0; i < _hw; i++)
                {
                    ptr[i] += coeff * output1[offset + i];
                }
            }
            for (int i = 0; i < _hw; i++)
            {
                ptr[i] = Sigmoid(ptr[i]);
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
            return 1.0f / (1.0f + MathF.Exp(-x));
        }
    }
}
