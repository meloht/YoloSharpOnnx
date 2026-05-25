using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Intrinsics.X86;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Linq;
using YoloSharpOnnx.Models;
using static System.Formats.Asn1.AsnWriter;


namespace YoloSharpOnnx.Inference.Classify
{
    internal class ClsPreprocess : PreprocessBase, IClsPreprocess
    {
        private readonly OnnxModel _onnxModel;
        private readonly YoloConfig _yoloConfig;
        public ClsPreprocess(OnnxModel onnxModel, YoloConfig yoloConfig)
        {
            _onnxModel = onnxModel;
            _yoloConfig = yoloConfig;
        }
        public void PreprocessImage(Mat inputImage, Mat resizedImg, FixedBuffer buffer)
        {
            // A.Resize: 将短边缩放到 targetSize
            int minSize = Math.Min(inputImage.Height, inputImage.Width);
            float scaleImg = (float)(_onnxModel.InputHeight) / minSize;
            int newW = _onnxModel.InputWidth;
            int newH = _onnxModel.InputHeight;

            if (inputImage.Height > inputImage.Width)
            {
                newH = (int)Math.Round(inputImage.Height * scaleImg);
            }
            else
            {
                newW = (int)Math.Round(inputImage.Width * scaleImg);
            }

            Cv2.Resize(inputImage, resizedImg, new OpenCvSharp.Size(newW, newH), interpolation: _yoloConfig.ResizeAlgorithm);

            //// B. CenterCrop: 从中心裁剪 224x224
            int startX = (newW - _onnxModel.InputWidth) / 2;
            int startY = (newH - _onnxModel.InputHeight) / 2;
            Rect roi = new Rect(startX, startY, _onnxModel.InputWidth, _onnxModel.InputHeight);
            using Mat cropped = new(resizedImg, roi);

            if (Avx2.IsSupported)
            {
                ToCHW_RGB_Normalized_AVX2(cropped, buffer);
            }
            else
            {
                ToCHW_RGB_Normalized(cropped, buffer);
            }

        }
    }
}
