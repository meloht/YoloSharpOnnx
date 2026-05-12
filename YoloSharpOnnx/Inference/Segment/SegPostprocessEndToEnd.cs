using Microsoft.ML.OnnxRuntime;
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
    public class SegPostprocessEndToEnd : ISegPostprocess
    {
        private readonly LabelModel[] _labels;
        private readonly YoloConfig _yoloConfig;
        public SegPostprocessEndToEnd(LabelModel[] labels, YoloConfig yoloConfig)
        {
            _labels = labels;
            _yoloConfig = yoloConfig;
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

                // 读取6个基础属性
                float x1 = (output0[offset + 0] - preResult.PadX) / preResult.Scale;
                float y1 = (output0[offset + 1] - preResult.PadY) / preResult.Scale;
                float x2 = (output0[offset + 2] - preResult.PadX) / preResult.Scale;
                float y2 = (output0[offset + 3] - preResult.PadY) / preResult.Scale;
                float score = output0[offset + 4];
                int classId = (int)output0[offset + 5];

                // 置信度过滤
                if (score < _yoloConfig.Confidence) continue;

                // 坐标裁剪到图像范围内
                x1 = Math.Max(0, Math.Min(x1, imgW));
                y1 = Math.Max(0, Math.Min(y1, imgH));
                x2 = Math.Max(0, Math.Min(x2, imgW));
                y2 = Math.Max(0, Math.Min(y2, imgH));

                Rect box = new Rect((int)x1, (int)y1, (int)(x2 - x1), (int)(y2 - y1));

                // ====================== 2. 读取 32 个掩码系数 ======================
                var coeffs = output0.Slice(offset + boxAttrs, maskCoeff);
                // ====================== 3. 解析 output1 原型掩码 [1,32,160,160] ======================

                float[] maskData = new float[protoH * protoW];

                // 矩阵乘法：coeffs(32) · protos(32, 160*160) → 160*160
                for (int y = 0; y < protoH; y++)
                {
                    for (int x = 0; x < protoW; x++)
                    {
                        float sum = 0;
                        for (int c = 0; c < maskCoeff; c++)
                        {
                            // output1 布局: [c, y, x]
                            sum += coeffs[c] * output1[c * protoH * protoW + y * protoW + x];
                        }
                        maskData[y * protoW + x] = Sigmoid(sum); // sigmoid激活
                    }
                }
                using Mat protoMask = new Mat(protoH, protoW, MatType.CV_32FC1);
                // 赋值到Mat
                protoMask.SetArray(maskData);

                // ====================== 4. 掩码缩放 + 二值化 ======================
                Mat finalMask = new Mat();
                Cv2.Resize(protoMask, finalMask, new Size(imgW, imgH));
                Cv2.Threshold(finalMask, finalMask, 0.5f, 1, ThresholdTypes.Binary);

                // 保存结果
                results.Add(new SegResult
                {
                    Box = box,
                    Confidence = score,
                    ClassId = classId,
                    ClassName = _labels[classId].Name,
                    Mask = finalMask
                });


            }

            return results;

        }

        private float Sigmoid(float x)
        {
            return 1.0f / (1.0f + (float)Math.Exp(-x));
        }
    }
}
