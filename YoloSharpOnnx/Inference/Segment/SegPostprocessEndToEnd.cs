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
using YoloSharpOnnx.Inference.OutputDecode;
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
            _classAtts = (int)onnx.OutputShape0[2];//38
            _boxAttrs = _classAtts - _maskDim;//38-32=6
        }

        public List<SegResult> PostProcess(OrtValue outputValue0, OrtValue outputValue1, PreDetectResult preResult)
        {
            List<SegResult> results = new List<SegResult>();

            var output0 = outputValue0.GetTensorDataAsSpan<float>();
            var output1 = outputValue1.GetTensorDataAsSpan<float>();
            using DisposableList<Mat> coeffMatList = new DisposableList<Mat>();
            // ====================== 1. 解析 output0 [1,300,38] ======================
            for (int i = 0; i < _maxDet; i++)
            {
                // 定位当前目标在数组中的起始位置
                int offset = i * _classAtts;

                float score = output0[offset + 4];

                // 置信度过滤
                if (score < _yoloConfig.Confidence) continue;

                int classId = (int)output0[offset + 5];
                // 读取6个基础属性
                Rect box = EndToEndDecode.Decode(output0, offset, preResult);

                var maskCoeffs = output0.Slice(offset + _boxAttrs, _maskDim);//maskCoeffs(32)
             
                coeffMatList.Add(GetCoeffMat(maskCoeffs));
               
                var result = new SegResult
                {
                    Box = box,
                    Confidence = score,
                    ClassId = classId,
                    ClassName = _labels[classId].Name
                };
                results.Add(result);
            }

            GEMM(results, coeffMatList, output1, preResult);
            

            return results;

        }


    }
}
