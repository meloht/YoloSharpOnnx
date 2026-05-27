using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Inference.Detect.Models;
using YoloSharpOnnx.Inference.DetectCore;
using YoloSharpOnnx.Inference.OutputDecode;
using YoloSharpOnnx.Models;

namespace YoloSharpOnnx.Inference.Obb
{
    internal class ObbPostprocessEndToEnd : IDetCorePostprocess<ObbResult>
    {
        private readonly LabelModel[] _labels;
        private readonly YoloConfig _yoloConfig;
        private readonly int _rowCount;
        private readonly int _colCount;
        public ObbPostprocessEndToEnd(OnnxModel onnx, YoloConfig yoloConfig)
        {
            _labels = onnx.Labels;
            _yoloConfig = yoloConfig;
            _rowCount = (int)onnx.OutputShape0[1];//[1,300,7]
            _colCount = (int)onnx.OutputShape0[2];//[1,300,7]
        }

        public void Dispose()
        {

        }
        private List<ObbResult> PostProcess(OrtValue outputValue, PreDetectResult preResult)
        {
            var detections = new List<ObbResult>();

            // 2. 使用 Span 直接访问内存，避免产生垃圾回收
            ReadOnlySpan<float> data = outputValue.GetTensorDataAsSpan<float>();

            for (int i = 0; i < _rowCount; i++)
            {
                // 计算当前行的偏移量
                int offset = i * _colCount;

                float confidence = data[offset + 4];

                // 过滤低置信度结果
                if (confidence < _yoloConfig.Confidence) continue;

                // 3. 提取坐标并还原到原始图像尺寸

                // 读取7个基础属性
                float cx = (data[offset + 0] - preResult.PadX) / preResult.Scale;
                float cy = (data[offset + 1] - preResult.PadY) / preResult.Scale;
                float w = data[offset + 2] / preResult.Scale;
                float h = data[offset + 3] / preResult.Scale;

                int labelId = (int)data[offset + 5];
                float angle = data[offset + 6];

                detections.Add(new ObbResult()
                {
                    Center = new Point2f(cx, cy),
                    Width = w,
                    Height = h,
                    Angle = YoloUtils.ToDegree(angle),
                    Confidence = confidence,
                    ClassId = labelId,
                    ClassName = _labels[labelId].Name
                });
            }

            return detections;
        }
        public List<ObbResult> PostProcessAsync(OrtValue output, PreDetectResult preResult)
        {
            return PostProcess(output, preResult);
        }

        public List<ObbResult> PostProcessSync(OrtValue output, PreDetectResult preResult)
        {
            return PostProcess(output, preResult);
        }
    }
}
