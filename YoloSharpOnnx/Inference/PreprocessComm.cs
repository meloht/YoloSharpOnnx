using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
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
           // Cv2.CvtColor(resizedImg, resizedImg, ColorConversionCodes.BGR2RGB);

            Cv2.CopyMakeBorder(
            src: resizedImg,
            dst: resizedImg,
            top: padH,        // 顶部填充
            bottom: _onnxModel.InputHeight - newImgH - padH, // 底部填充（补全到 1280）
            left: padW,       // 左侧填充
            right: _onnxModel.InputWidth - newImgW - padW,  // 右侧填充（补全到 1280）
            borderType: BorderTypes.Constant,
            value: _paddingColor);



            //GetChwArrPointer(resizedImg, buffer);
            if (Avx2.IsSupported)
            {
                ToCHW_RGB_Normalized_AVX2(resizedImg, buffer);
            }
            else
            {
                ToCHW_RGB_Normalized(resizedImg, buffer);
            }
          
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

        private unsafe void ToCHW_RGB_Normalized(Mat mat, FixedBuffer buffer)
        {
            int width = mat.Cols;
            int height = mat.Rows;
            int channels = mat.Channels(); 

            if (channels != 3)
                throw new ArgumentException("Only 3-channel images supported");

            byte* ptr = (byte*)mat.DataPointer;
            float* data = buffer.Pointer;
            int hw = width * height;

            // 三个通道分开写（CHW）
            int rOffset = 0;
            int gOffset = hw;
            int bOffset = hw * 2;
            float scale = 1.0f / 255.0f;

            for (int y = 0; y < height; y++)
            {
                int rowOffset = y * width * channels;

                for (int x = 0; x < width; x++)
                {
                    int srcIndex = rowOffset + x * channels;

                    int dstIndex = y * width + x;

                    //  BGR -> RGB + 归一化 + CHW
                    data[rOffset + dstIndex] = ptr[srcIndex + 2] * scale;
                    data[gOffset + dstIndex] = ptr[srcIndex + 1] * scale;
                    data[bOffset + dstIndex] = ptr[srcIndex + 0] * scale;
                }
            }
        }

        public static unsafe void ToCHW_RGB_Normalized_AVX2(Mat mat, FixedBuffer buffer)
        {
            if (!Avx2.IsSupported)
                throw new NotSupportedException("AVX2 not supported");

            int width = mat.Width;
            int height = mat.Height;
            byte* src= (byte*)mat.DataPointer;
            float* dst = buffer.Pointer;
            int hw =width *height;

            float* dstR = dst;
            float* dstG = dst + hw;
            float* dstB = dst + hw * 2;
            float scale1 = 1.0f / 255.0f;

            Vector256<float> scale = Vector256.Create(1.0f / 255.0f);

            int stride = width * 3;

            for (int y = 0; y < height; y++)
            {
                byte* row = src + y * stride;

                int x = 0;

                // 每次处理 8 像素（24 字节）
                for (; x <= width - 8; x += 8)
                {
                    byte* p = row + x * 3;

                    // 手动加载（因为不是对齐的）
                    uint b0 = p[0]; uint g0 = p[1]; uint r0 = p[2];
                    uint b1 = p[3]; uint g1 = p[4]; uint r1 = p[5];
                    uint b2 = p[6]; uint g2 = p[7]; uint r2 = p[8];
                    uint b3 = p[9]; uint g3 = p[10]; uint r3 = p[11];
                    uint b4 = p[12]; uint g4 = p[13]; uint r4 = p[14];
                    uint b5 = p[15]; uint g5 = p[16]; uint r5 = p[17];
                    uint b6 = p[18]; uint g6 = p[19]; uint r6 = p[20];
                    uint b7 = p[21]; uint g7 = p[22]; uint r7 = p[23];

                    // 构建向量（R）
                    var vr = Vector256.Create(
                        (float)r0, (float)r1, (float)r2, (float)r3,
                        (float)r4, (float)r5, (float)r6, (float)r7);

                    var vg = Vector256.Create(
                        (float)g0, (float)g1, (float)g2, (float)g3,
                        (float)g4, (float)g5, (float)g6, (float)g7);

                    var vb = Vector256.Create(
                        (float)b0, (float)b1, (float)b2, (float)b3,
                        (float)b4, (float)b5, (float)b6, (float)b7);

                    // 归一化
                    vr = Avx.Multiply(vr, scale);
                    vg = Avx.Multiply(vg, scale);
                    vb = Avx.Multiply(vb, scale);

                    int idx = y * width + x;

                    Avx.Store(dstR + idx, vr);
                    Avx.Store(dstG + idx, vg);
                    Avx.Store(dstB + idx, vb);
                }

                // 处理尾部
                for (; x < width; x++)
                {
                    byte* p = row + x * 3;

                    int idx = y * width + x;

                    dstR[idx] = p[2] * scale1;
                    dstG[idx] = p[1] * scale1;
                    dstB[idx] = p[0] * scale1;
                }
            }
        }
    }
}
