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
using YoloSharpOnnx.Inference.DetectCore;
using YoloSharpOnnx.Inference.OutputDecode;
using YoloSharpOnnx.Models;

namespace YoloSharpOnnx.Inference.Pose
{
    internal class PosePostprocessNMS : IDetCorePostprocess<PoseResult>
    {
        private readonly LabelModel[] _labels;
        private readonly NmsDecode _nmsDecode;
        private readonly List<Rect> _boxes = new List<Rect>();
        private readonly List<float> _scores = new List<float>();
        private readonly List<int> _classIds = new List<int>();
        private readonly List<int> _ids = new List<int>();
        private readonly Lazy<ObjectPool<PostResultArray>> _postResultPool;

        private readonly int _numAnchors;
        private readonly int _kCount;
        private readonly int _kDim;

        public PosePostprocessNMS(OnnxModel onnx, YoloConfig yoloConfig)
        {
            _labels = onnx.Labels;
            _nmsDecode = new NmsDecode(onnx, yoloConfig);
            _kCount = onnx.KPTShape[0];//[17, 3]
            _kDim = onnx.KPTShape[1];
            _numAnchors = (int)onnx.OutputShape0[2];//[1,56,8400]
            _postResultPool = new Lazy<ObjectPool<PostResultArray>>(() => new ObjectPool<PostResultArray>(PostResultArray.CreateForSegment, yoloConfig.BatchPoolSize, YoloUtils.ClearList));
        }

        private List<PoseResult> PostProcessBase(OrtValue outputValue, PreDetectResult preResult, List<Rect> boxes, List<float> scores, List<int> classIds, List<int> ids)
        {

            var ortSpan = outputValue.GetTensorDataAsSpan<float>();//[1,56,8400]

            int[] indices = _nmsDecode.Decode(ortSpan, preResult, boxes, scores, classIds, ids);

            List<PoseResult> results = new List<PoseResult>();
            // 绘制检测结果
            foreach (var idx in indices)
            {
                Rect box = boxes[idx];
                float score = scores[idx];
                int class_id = classIds[idx];
                string lable = _labels[class_id].Name;

                PosePoint[] posePoints = new PosePoint[_kCount];

                for (int k = 0; k < _kCount; k++)
                {
                    int kpBase = 5 + k * _kDim;// cx,cy,w,h,confidence  keypoints:[x,y,score]  5 + k*3

                    float kx = ortSpan[(kpBase + 0) * _numAnchors + ids[idx]];
                    float ky = ortSpan[(kpBase + 1) * _numAnchors + ids[idx]];
                    float ks = ortSpan[(kpBase + 2) * _numAnchors + ids[idx]];

                    kx = (kx - preResult.PadX) / preResult.Scale;
                    ky = (ky - preResult.PadY) / preResult.Scale;

                    posePoints[k] = new PosePoint(kx, ky, k, ks);

                }

                PoseResult detection = new PoseResult();
                detection.Confidence = score;
                detection.ClassName = lable;
                detection.ClassId = class_id;
                detection.Box = box;
                detection.KeyPoints = posePoints;
                results.Add(detection);

            }

            return results;
        }


        public void Dispose()
        {
            if (_postResultPool.IsValueCreated)
            {
                _postResultPool.Value.Dispose();
            }
        }

        public List<PoseResult> PostProcessAsync(OrtValue outputValue0, PreDetectResult preResult)
        {
            var arr = _postResultPool.Value.Rent();
            try
            {
                return PostProcessBase(outputValue0, preResult, arr.Boxes, arr.Scores, arr.ClassIds, arr.Ids);
            }
            finally
            {
                _postResultPool.Value.Return(arr);
            }
        }

        public List<PoseResult> PostProcessSync(OrtValue outputValue0, PreDetectResult preResult)
        {
            _boxes.Clear();
            _scores.Clear();
            _classIds.Clear();
            _ids.Clear();
            return PostProcessBase(outputValue0, preResult, _boxes, _scores, _classIds, _ids);
        }
    }
}
