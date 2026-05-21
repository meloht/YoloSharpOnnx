using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;
using OpenCvSharp.Dnn;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Inference.Detect.Models;
using YoloSharpOnnx.Inference.OutputDecode;
using YoloSharpOnnx.Models;

namespace YoloSharpOnnx.Inference.Pose
{
    public class PosePostprocessEndToEnd : IPosePostprocess
    {
        private readonly LabelModel[] _labels;
        private readonly YoloConfig _yoloConfig;
        private readonly int _rowCount;
        private readonly int _colCount;
        private readonly int _kCount;
        private readonly int _kDim;

        public PosePostprocessEndToEnd(OnnxModel onnx, YoloConfig yoloConfig)
        {
            _labels = onnx.Labels;
            _yoloConfig = yoloConfig;
            _rowCount = (int)onnx.OutputShape0[1];//[1,300,57]
            _colCount = (int)onnx.OutputShape0[2];//[1,300,57] 4 + 1 + 1 + 51 = 57  x1, y1, x2, y2,score
            _kCount = onnx.KPTShape[0];//[17, 3]
            _kDim = onnx.KPTShape[1];
        }
        private List<PoseResult> PostProcess(OrtValue outputValue, PreDetectResult preResult)
        {
            var detections = new List<PoseResult>();

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

                // 读取6个基础属性
                Rect box = EndToEndDecode.Decode(data, offset, preResult);

                int labelId = (int)data[offset + 5];

                int kpBase = offset + 6;

                PosePoint[] posePoints = new PosePoint[_kCount];
                for (int k = 0; k < _kCount; k++)
                {
                    float x = data[kpBase + k * _kDim + 0];
                    float y = data[kpBase + k * _kDim + 1];
                    float s = data[kpBase + k * _kDim + 2];

                    // letterbox reverse
                    x = (x - preResult.PadX) / preResult.Scale;
                    y = (y - preResult.PadY) / preResult.Scale;

                    posePoints[k] = new PosePoint(x, y, k, s);
                }

                detections.Add(new PoseResult()
                {
                    Box = box,
                    Confidence = confidence,
                    ClassId = labelId,
                    ClassName = _labels[labelId].Name,
                    KeyPoints = posePoints
                });
            }

            return detections;
        }

        public List<PoseResult> PostProcessAsync(OrtValue outputValue0, PreDetectResult preResult)
        {
            return PostProcess(outputValue0, preResult);
        }

        public List<PoseResult> PostProcessSync(OrtValue outputValue0, PreDetectResult preResult)
        {
            return PostProcess(outputValue0, preResult);
        }

        public void Dispose()
        {

        }
    }
}
