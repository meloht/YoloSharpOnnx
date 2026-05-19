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


namespace YoloSharpOnnx.Inference.Segment
{
    public class SegPostprocessNMS : SegPostprocessBase, ISegPostprocess
    {
        private readonly int _numAnchors;
        private readonly int _classAtts;
        private readonly NmsDecode _nmsDecode;

        private readonly List<Rect> _boxes = new List<Rect>();
        private readonly List<float> _scores = new List<float>();
        private readonly List<int> _classIds = new List<int>();
        private readonly List<int> _ids = new List<int>();

        private readonly Lazy<ObjectPool<PostResultArray>> _postResultPool;


        public SegPostprocessNMS(OnnxModel onnx, YoloConfig yoloConfig) : base(onnx, yoloConfig)
        {
            _numAnchors = (int)onnx.OutputShape0[2];
            _classAtts = (int)onnx.OutputShape0[1] - _maskDim;//[1,116,8400] 116-32=84
            _nmsDecode = new NmsDecode(onnx, yoloConfig);
            _postResultPool = new Lazy<ObjectPool<PostResultArray>>(() => new ObjectPool<PostResultArray>(PostResultArray.CreateForSegment,yoloConfig.BatchPoolSize, ClearList));

        }
        private void ClearList(PostResultArray resultArray)
        {
            resultArray.Boxes.Clear();
            resultArray.Scores.Clear();
            resultArray.ClassIds.Clear();
            resultArray.Ids.Clear();
        }
        private List<SegResult> PostProcessBase(OrtValue outputValue0, OrtValue outputValue1, PreDetectResult preResult,
            List<Rect> boxes, List<float> scores, List<int> classIds, List<int> ids, YoloSegDecode segDecode)
        {

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
            segDecode.Decode(results, coeffMatList, output1, preResult);
            return results;
        }

        public List<SegResult> PostProcessAsync(OrtValue outputValue0, OrtValue outputValue1, PreDetectResult preResult)
        {

            var decode = _segDecodePool.Value.Rent();
            var arr= _postResultPool.Value.Rent();
            try
            {
                return PostProcessBase(outputValue0, outputValue1, preResult, arr.Boxes, arr.Scores, arr.ClassIds, arr.Ids, decode);
            }
            finally
            {
                _segDecodePool.Value.Return(decode);
                _postResultPool.Value.Return(arr);
            }

        }

        public List<SegResult> PostProcessSync(OrtValue outputValue0, OrtValue outputValue1, PreDetectResult preResult)
        {
            _boxes.Clear();
            _scores.Clear();
            _classIds.Clear();
            _ids.Clear();
            return PostProcessBase(outputValue0, outputValue1, preResult, _boxes, _scores, _classIds, _ids, _yoloSegDecode);
        }

        public void Dispose()
        {
            DisposeBase();
            if (_postResultPool.IsValueCreated)
            {
                _postResultPool.Value.Dispose();
            }
        }
    }
}
