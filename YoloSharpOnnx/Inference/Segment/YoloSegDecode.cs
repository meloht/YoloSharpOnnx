using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Inference.Detect.Models;
using YoloSharpOnnx.Models;

namespace YoloSharpOnnx.Inference.Segment
{
    public class YoloSegDecode : IDisposable
    {
        private readonly int _maskDim;
        private readonly int _protoH;
        private readonly int _protoW;

        private readonly int _inputW;
        private readonly int _inputH;
        private readonly int _protoSize;

        protected const float _threshold = 0.5f;

        // ===== 只保留3个 buffer（极致优化）=====
        private readonly Mat _coeffMat = new();
        private readonly Mat _protoMat = new();
        private readonly Mat _maskMat = new();
        private readonly Mat _outMat = new();

        public YoloSegDecode(OnnxModel onnx, YoloConfig yoloConfig)
        {
            _inputH = onnx.InputHeight;
            _inputW = onnx.InputWidth;
            _protoH = (int)onnx.OutputShape1[2];// [1,32,160,160] 160
            _protoW = (int)onnx.OutputShape1[3];//[1,32,160,160] 160
            _maskDim = (int)onnx.OutputShape1[1];//[1,32,160,160]  32 

            _protoSize = _protoH * _protoW;
        }

        // =========================================================
        // ZERO-COPY GEMM + DECODE
        // =========================================================
        public unsafe void Decode(List<SegResult> list, List<Mat> coeffMatList, ReadOnlySpan<float> proto, PreDetectResult pre)
        {
            int n = list.Count;
            if (n == 0) return;

            // 1. coeffMat (copy only ONCE, no extra Mat)
            _coeffMat.Create(n, _maskDim, MatType.CV_32FC1);

            // DataPointer is already an unmanaged pointer; no need to use fixed here.
            float* dstBase = (float*)_coeffMat.DataPointer;
            for (int i = 0; i < n; i++)
            {
                float* src = (float*)coeffMatList[i].DataPointer;
                float* dst = dstBase + i * _maskDim;
                Buffer.MemoryCopy(src, dst, _maskDim * sizeof(float), _maskDim * sizeof(float));
            }

            // 2. protoMat (NO COPY if already contiguous Span)
            _protoMat.Create(_maskDim, _protoSize, MatType.CV_32FC1);

            // DataPointer is already an unmanaged pointer; pin only the span `proto`.
            float* dstProto = (float*)_protoMat.DataPointer;
            fixed (float* src = proto)
            {
                Buffer.MemoryCopy(src, dstProto, proto.Length * sizeof(float), proto.Length * sizeof(float));
            }

            // =====================================================
            // 3. GEMM
            // =====================================================
            Cv2.Gemm(_coeffMat, _protoMat, 1.0, _outMat, 0.0, _maskMat);

            // =====================================================
            // 4. sigmoid (in-place, no temp Mat)
            // =====================================================
            Cv2.Multiply(_maskMat, -1.0, _maskMat);
            Cv2.Exp(_maskMat, _maskMat);
            Cv2.Add(_maskMat, 1.0, _maskMat);
            Cv2.Divide(1.0, _maskMat, _maskMat);

            // =====================================================
            // 5. ZERO-COPY mask view (关键)
            // =====================================================
            float* basePtr = (float*)_maskMat.DataPointer;

            for (int i = 0; i < n; i++)
            {
                float* maskPtr = basePtr + i * _protoSize;

                // zero-copy view
                Mat mask = Mat.FromPixelData(_protoH, _protoW, MatType.CV_32FC1, (IntPtr)maskPtr);
                // =================================================
                // 6. ONLY ONE resize (final stage)
                // =================================================
                Cv2.Resize(mask, mask, new Size(_inputW, _inputH), interpolation: InterpolationFlags.Linear);

                // =================================================
                // 7. letterbox restore (zero-copy ROI)
                // =================================================
                int left = (int)pre.PadX;
                int top = (int)pre.PadY;

                int w = (int)(pre.ImageWidth * pre.Scale);
                int h = (int)(pre.ImageHeight * pre.Scale);

                Rect roi = new Rect(left, top, Math.Min(w, mask.Width - left), Math.Min(h, mask.Height - top));

                using Mat cropped = new Mat(mask, roi);

                // =================================================
                // 8. final resize (only once)
                // =================================================

                Cv2.Resize(cropped, cropped, new Size(pre.ImageWidth, pre.ImageHeight), interpolation: InterpolationFlags.Linear);

                // =================================================
                // 9. threshold (no extra Mat)
                // =================================================
                Cv2.Threshold(cropped, cropped, _threshold, 255, ThresholdTypes.Binary);
                cropped.ConvertTo(cropped, MatType.CV_8UC1);

                // =================================================
                // 10. ROI mask (ZERO COPY FINAL)
                // =================================================

                list[i].Mask = new Mat(cropped, list[i].Box);
            }
        }

        public void Dispose()
        {
            _coeffMat.Dispose();
            _protoMat.Dispose();
            _maskMat.Dispose();
            _outMat.Dispose();
        }
    }
}
