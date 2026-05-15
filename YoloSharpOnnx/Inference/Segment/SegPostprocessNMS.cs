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

        private readonly List<Rect> _boxes = new List<Rect>();
        private readonly List<float> _scores = new List<float>();
        private readonly List<int> _classIds = new List<int>();
        private readonly List<int> _ids = new List<int>();

        private readonly int _classAtts;

        private readonly NmsDecode _nmsDecode;


        public SegPostprocessNMS(OnnxModel onnx, YoloConfig yoloConfig) : base(onnx, yoloConfig)
        {
            _numAnchors = (int)onnx.OutputShape0[2];
            _classAtts = (int)onnx.OutputShape0[1] - _maskDim;//[1,116,8400] 116-32=84
            _nmsDecode = new NmsDecode(onnx, yoloConfig, _boxes, _scores, _classIds, _ids);

        }
        public List<SegResult> PostProcess(OrtValue outputValue0, OrtValue outputValue1, PreDetectResult preResult)
        {
            _boxes.Clear();
            _scores.Clear();
            _classIds.Clear();
            _ids.Clear();

            var output0 = outputValue0.GetTensorDataAsSpan<float>();
            var output1 = outputValue1.GetTensorDataAsSpan<float>();

            int[] indices = _nmsDecode.Decode(output0, preResult);
           
            List<SegResult> results = new List<SegResult>();
            using DisposableList<Mat> coeffMatList = new DisposableList<Mat>();

            foreach (var idx in indices)
            {
                float[] maskCoeffs = new float[32];
                for (int m = 0; m < _maskDim; m++)
                {
                    maskCoeffs[m] = output0[(_classAtts + m) * _numAnchors + _ids[idx]];
                }
                var result = new SegResult
                {
                    Box = _boxes[idx],
                    Confidence = _scores[idx],
                    ClassId = _classIds[idx],
                    ClassName = _labels[_classIds[idx]].Name
                };
                results.Add(result);
            }
            GEMM(results, coeffMatList, output1, preResult);
            return results;
        }
    }
}
