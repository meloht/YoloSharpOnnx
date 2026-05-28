using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Text;
using System.Threading.Tasks;
using YoloSharpOnnx.Utils;

namespace YoloSharpOnnx.Inference
{
    internal class PreprocessBase
    {
        protected unsafe void ToCHW_RGB_Normalized(Mat mat, FixedBuffer buffer)
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
            long step = mat.Step();
            for (int y = 0; y < height; y++)
            {
                byte* rowPtr = ptr + y * step;
                int rowOffset = y * width;

                for (int x = 0; x < width; x++)
                {

                    byte* pixel = rowPtr + x * channels;
                    byte b = pixel[0];
                    byte g = pixel[1];
                    byte r = pixel[2];

                    int idx = rowOffset + x;

                    //  BGR -> RGB + 归一化 + CHW
                    data[rOffset + idx] = r * scale;
                    data[gOffset + idx] = g * scale;
                    data[bOffset + idx] = b * scale;
                }
            }
        }

        protected unsafe void ToCHW_RGB_Normalized_AVX2(Mat mat, FixedBuffer buffer)
        {
            if (!Avx2.IsSupported)
                throw new NotSupportedException("AVX2 not supported");

            int width = mat.Width;
            int height = mat.Height;
            byte* src = (byte*)mat.DataPointer;
            float* dst = buffer.Pointer;
            int hw = width * height;

            float* dstR = dst;
            float* dstG = dst + hw;
            float* dstB = dst + hw * 2;
            float inv255 = 1.0f / 255.0f;

            Vector256<float> scale = Vector256.Create(inv255);
            int step = (int)mat.Step();

            int x = 0;
            for (int y = 0; y < height; y++)
            {
                byte* row = src + y * step;

                x = 0;

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

                    dstR[idx] = p[2] * inv255;
                    dstG[idx] = p[1] * inv255;
                    dstB[idx] = p[0] * inv255;
                }
            }
        }
    }
}
