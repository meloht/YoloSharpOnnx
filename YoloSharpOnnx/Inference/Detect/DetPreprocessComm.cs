using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Text;
using System.Threading.Tasks;
using YoloSharpOnnx.Inference.Detect.Models;
using YoloSharpOnnx.Models;
using YoloSharpOnnx.Utils;

namespace YoloSharpOnnx.Inference.Detect
{
    internal class DetPreprocessComm : PreprocessBase, IDetPreprocess
    {
        protected readonly Scalar _paddingColor;
        private readonly OnnxModel _onnxModel;
        private readonly YoloConfig _yoloConfig;

        public DetPreprocessComm(OnnxModel onnxModel, YoloConfig yoloConfig)
        {
            _onnxModel = onnxModel;
            _paddingColor = new Scalar(114, 114, 114);
            _yoloConfig = yoloConfig;
        }
        public PreDetectResult PreprocessImage(Mat inputImage,Mat resizedImg, FixedBuffer buffer)
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
            float padW = (float)(_onnxModel.InputWidth - newImgW) / 2.0f; // 左右填充的一半
            float padH = (float)(_onnxModel.InputHeight - newImgH) / 2.0f; // 上下填充的一半


            // 5. 缩放图像（若原始尺寸≠缩放后尺寸）
            Cv2.Resize(inputImage, resizedImg, new OpenCvSharp.Size(newImgW, newImgH), interpolation: _yoloConfig.ResizeAlgorithm);


            int top = (int)Math.Round(padH - 0.1);
            int bottom = (int)Math.Round(padH + 0.1);
            int left = (int)Math.Round(padW - 0.1);
            int right = (int)Math.Round(padW + 0.1);

            // BGR转RGB
            // Cv2.CvtColor(resizedImg, resizedImg, ColorConversionCodes.BGR2RGB);

            Cv2.CopyMakeBorder(
            src: resizedImg,
            dst: resizedImg,
            top: top,        // 顶部填充
            bottom: bottom, // 底部填充（补全到 1280）
            left: left,       // 左侧填充
            right: right,  // 右侧填充（补全到 1280）
            borderType: BorderTypes.Constant,
            value: _paddingColor);

            if (Avx2.IsSupported)
            {
                ToCHW_RGB_Normalized_AVX2(resizedImg, buffer);
            }
            else
            {
                ToCHW_RGB_Normalized(resizedImg, buffer);
            }

            // 添加批次维度 (1, 3, H, W)
            return new PreDetectResult(imgH, imgW, top, left, scale);
        }


    }
}
