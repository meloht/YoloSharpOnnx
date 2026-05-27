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
using YoloSharpOnnx.Inference.Obb.Models;
using YoloSharpOnnx.Inference.OutputDecode;
using YoloSharpOnnx.Models;

namespace YoloSharpOnnx.Inference.Obb
{
    internal class ObbPostprocessNMS : IDetCorePostprocess<ObbResult>
    {
        private readonly LabelModel[] _labels;
        private readonly NmsDecode _nmsDecode;
        private readonly List<ObbResult> _boxes = new List<ObbResult>();

        private readonly Lazy<ObjectPool<ObbList>> _postResultPool;

        private readonly int _numAnchors;
        public ObbPostprocessNMS(OnnxModel onnx, YoloConfig yoloConfig)
        {
            _labels = onnx.Labels;
            _nmsDecode = new NmsDecode(onnx, yoloConfig);
            _numAnchors = (int)onnx.OutputShape0[2];//[1,20,21504]
            _postResultPool = new Lazy<ObjectPool<ObbList>>(() => new ObjectPool<ObbList>(() => new ObbList(), yoloConfig.BatchPoolSize, ClearList));
        }

        public void Dispose()
        {
            if (_postResultPool.IsValueCreated)
            {
                _postResultPool.Value.Dispose();
            }
        }
        private void ClearList(ObbList obbList)
        {
            obbList.Results?.Clear();
        }

        private List<ObbResult> PostProcessBase(OrtValue outputValue, PreDetectResult preResult, List<ObbResult> boxes)
        {
            var ortSpan = outputValue.GetTensorDataAsSpan<float>();//[1,56,8400]
            List<ObbResult> results = _nmsDecode.Decode(ortSpan, preResult, boxes, _labels);
            return results;
        }


        public List<ObbResult> PostProcessAsync(OrtValue output, PreDetectResult preResult)
        {
            var arr = _postResultPool.Value.Rent();
            try
            {
                return PostProcessBase(output, preResult, arr.Results);
            }
            finally
            {
                _postResultPool.Value.Return(arr);
            }
        }

        public List<ObbResult> PostProcessSync(OrtValue output, PreDetectResult preResult)
        {
            _boxes.Clear();
            return PostProcessBase(output, preResult, _boxes);
        }
    }
}
