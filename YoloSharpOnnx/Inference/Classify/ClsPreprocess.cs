using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Intrinsics.X86;
using System.Text;
using System.Threading.Tasks;
using YoloSharpOnnx.Models;


namespace YoloSharpOnnx.Inference.Classify
{
    public class ClsPreprocess : PreprocessBase, IClsPreprocess
    {
        private readonly OnnxModel _onnxModel;
        public ClsPreprocess(OnnxModel onnxModel)
        {
            _onnxModel = onnxModel;
        }
        public void PreprocessImage(Mat inputImage, Mat resizedImg, FixedBuffer buffer, InterpolationFlags interpolationFlags)
        {
            // A. Resize: 将短边缩放到 targetSize
            int minSize = Math.Min(inputImage.Height, inputImage.Width);
            float scaleImg = (float)_onnxModel.InputHeight / minSize;
            int newW = _onnxModel.InputWidth;
            int newH = _onnxModel.InputHeight;

            if (inputImage.Height > inputImage.Width)
            {
                newH = (int)(inputImage.Height * scaleImg);
            }
            else
            {
                newW = (int)(inputImage.Width * scaleImg);
            }

            Cv2.Resize(inputImage, resizedImg, new OpenCvSharp.Size(newW, newH), interpolation: interpolationFlags);

            //// B. CenterCrop: 从中心裁剪 224x224
            int startX = (newW - _onnxModel.InputWidth) / 2;
            int startY = (newH - _onnxModel.InputHeight) / 2;
            Rect roi = new Rect(startX, startY, _onnxModel.InputWidth, _onnxModel.InputHeight);
            using Mat cropped = new Mat(resizedImg, roi);

            if (Avx2.IsSupported)
            {
                ToCHW_RGB_Normalized_AVX2(resizedImg, buffer);
            }
            else
            {
                ToCHW_RGB_Normalized(resizedImg, buffer);
            }
        }


    }
}
