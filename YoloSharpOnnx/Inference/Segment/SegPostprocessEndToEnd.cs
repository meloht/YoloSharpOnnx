using Microsoft.ML.OnnxRuntime;
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
            throw new NotImplementedException();
        }
    }
}
