using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;
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
    public class SegPostprocessNMS : ISegPostprocess
    {
        private readonly int _boxNums;
        private readonly int _boxNums2;
        private readonly int _boxNums3;
        private readonly int _boxNums4;
        private readonly LabelModel[] _labels;


        private List<Rect> _boxes = new List<Rect>();
        private List<float> _scores = new List<float>();
        private List<int> _classIds = new List<int>();
        private readonly YoloConfig _yoloConfig;

        public SegPostprocessNMS(int boxNum, LabelModel[] labels, YoloConfig yoloConfig)
        {
            _labels = labels;
            _boxNums = boxNum;
            _boxNums2 = _boxNums * 2;
            _boxNums3 = _boxNums * 3;
            _boxNums4 = _boxNums * 4;
            _yoloConfig = yoloConfig;
        }
        public List<SegResult> PostProcess(OrtValue outputValue0, OrtValue outputValue1, PreDetectResult preResult)
        {
            throw new NotImplementedException();
        }
    }
}
