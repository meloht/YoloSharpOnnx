using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;
using OpenCvSharp.Dnn;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Inference.Detect.Models;
using YoloSharpOnnx.Models;

namespace YoloSharpOnnx.Inference.Segment
{
    public class SegPostprocessEndToEnd : SegPostprocessBase, ISegPostprocess
    {
        private readonly int _maxDet;
        private readonly int _classAtts;
        private readonly int _boxAttrs;
      
        public SegPostprocessEndToEnd(OnnxModel onnx, YoloConfig yoloConfig) : base(onnx, yoloConfig)
        {
            _maxDet = (int)onnx.OutputShape0[1]; //[1,300,38]   300
            _classAtts= (int)onnx.OutputShape0[2];//38
            _boxAttrs = _classAtts - _maskDim;//38-32=6
        }

        public List<SegResult> PostProcess(OrtValue outputValue0, OrtValue outputValue1, PreDetectResult preResult)
        {
            List<SegResult> results = new List<SegResult>();

            var output0 = outputValue0.GetTensorDataAsSpan<float>();
            var output1 = outputValue1.GetTensorDataAsSpan<float>();

            // ====================== 1. 解析 output0 [1,300,38] ======================
            for (int i = 0; i < _maxDet; i++)
            {
                // 定位当前目标在数组中的起始位置
                int offset = i * _classAtts;

                float score = output0[offset + 4];

                // 置信度过滤
                if (score < _yoloConfig.Confidence) continue;

                // 读取6个基础属性
                float x1 = (output0[offset + 0] - preResult.PadX) / preResult.Scale;
                float y1 = (output0[offset + 1] - preResult.PadY) / preResult.Scale;
                float x2 = (output0[offset + 2] - preResult.PadX) / preResult.Scale;
                float y2 = (output0[offset + 3] - preResult.PadY) / preResult.Scale;
                int classId = (int)output0[offset + 5];

                Rect box = new Rect((int)x1, (int)y1, (int)(x2 - x1), (int)(y2 - y1));

                var maskCoeffs = output0.Slice(offset + _boxAttrs, _maskDim);//maskCoeffs(32)

                SegResult result = BuildResult(box, classId, score, maskCoeffs, protoH, protoW, output1, preResult);

                results.Add(result);
            }

            return results;

        }


    }
}
