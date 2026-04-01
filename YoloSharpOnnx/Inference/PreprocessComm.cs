using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YoloSharpOnnx.Models;

namespace YoloSharpOnnx.Inference
{
    public class PreprocessComm : IPreprocess
    {
        protected readonly Scalar _paddingColor;
        private readonly OnnxModel _onnxModel;

        public PreprocessComm(OnnxModel onnxModel)
        {
            _onnxModel = onnxModel;
            _paddingColor = new Scalar(114, 114, 114);
        }
        public PreResult PreprocessImage(Mat inputImage, Mat resizedImg, FixedBuffer buffer, InterpolationFlags interpolationFlags)
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
            int padW = (_onnxModel.InputWidth - newImgW) / 2; // 左右填充的一半
            int padH = (_onnxModel.InputHeight - newImgH) / 2; // 上下填充的一半


            // 5. 缩放图像（若原始尺寸≠缩放后尺寸）

            Cv2.Resize(inputImage, resizedImg, new OpenCvSharp.Size(newImgW, newImgH), interpolation: interpolationFlags);

            // BGR转RGB
            Cv2.CvtColor(resizedImg, resizedImg, ColorConversionCodes.BGR2RGB);

            Cv2.CopyMakeBorder(
               src: resizedImg,
               dst: resizedImg,
               top: padH,        // 顶部填充
               bottom: _onnxModel.InputHeight - newImgH - padH, // 底部填充（补全到 1280）
               left: padW,       // 左侧填充
               right: _onnxModel.InputWidth - newImgW - padW,  // 右侧填充（补全到 1280）
               borderType: BorderTypes.Constant,
               value: _paddingColor // 填充色（BGR 格式）
           );

            GetChwArrPointer(resizedImg, buffer);

            // 添加批次维度 (1, 3, H, W)
            return new PreResult(imgH, imgW, padH, padW, scale);
        }
        public unsafe void GetChwArrPointer(Mat paddedImg, FixedBuffer buffer)
        {
            int height = paddedImg.Height;
            int width = paddedImg.Width;
            
            float inv255 = 1.0f / 255.0f;

            byte* ptr = (byte*)paddedImg.DataPointer;
            float* data = buffer.Pointer;
            int hw = width * height;

            for (int i = 0; i < hw; i++)
            {
                data[i] = ptr[i * 3 + 0] * inv255;           // R
                data[i + hw] = ptr[i * 3 + 1] * inv255;      // G
                data[i + hw * 2] = ptr[i * 3 + 2] * inv255;  // B
            }

        }
    }
}
