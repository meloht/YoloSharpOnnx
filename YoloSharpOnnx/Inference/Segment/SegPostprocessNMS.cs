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
using static System.Formats.Asn1.AsnWriter;

namespace YoloSharpOnnx.Inference.Segment
{
    public class SegPostprocessNMS : SegPostprocessBase, ISegPostprocess
    {
        private readonly int _numAnchors;



        private readonly int _classAtts;

        private readonly NmsDecode _nmsDecode;


        public SegPostprocessNMS(OnnxModel onnx, YoloConfig yoloConfig) : base(onnx, yoloConfig)
        {
            _numAnchors = (int)onnx.OutputShape0[2];
            _classAtts = (int)onnx.OutputShape0[1] - _maskDim;//[1,116,8400] 116-32=84
            _nmsDecode = new NmsDecode(onnx, yoloConfig);

        }
        public List<SegResult> PostProcess(OrtValue outputValue0, OrtValue outputValue1, PreDetectResult preResult)
        {
            List<Rect> boxes = new List<Rect>();
            List<float> scores = new List<float>();
            List<int> classIds = new List<int>();
            List<int> ids = new List<int>();

            var output0 = outputValue0.GetTensorDataAsSpan<float>();
            var output1 = outputValue1.GetTensorDataAsSpan<float>();

            int[] indices = _nmsDecode.Decode(output0, preResult, boxes, scores, classIds, ids);
           
            List<SegResult> results = new List<SegResult>();
            using DisposableList<Mat> coeffMatList = new DisposableList<Mat>();

            foreach (var idx in indices)
            {
                Mat coeffMat = new Mat(1, _maskDim, MatType.CV_32FC1);
                unsafe
                {
                    float* coeffPtr = (float*)coeffMat.DataPointer;
                    for (int m = 0; m < _maskDim; m++)
                    {
                        coeffPtr[m] = output0[(_classAtts + m) * _numAnchors + ids[idx]];
                    }
                }

                coeffMatList.Add(coeffMat);
                var result = new SegResult
                {
                    Box = boxes[idx],
                    Confidence = scores[idx],
                    ClassId = classIds[idx],
                    ClassName = _labels[classIds[idx]].Name
                   
                };
                results.Add(result);
            }
            GEMM(results, coeffMatList, output1, preResult);
            return results;
        }
    }
}
