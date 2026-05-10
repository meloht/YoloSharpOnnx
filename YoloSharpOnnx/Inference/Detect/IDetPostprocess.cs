using Microsoft.ML.OnnxRuntime;
using System;
using System.Collections.Generic;
using System.Text;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Inference.Detect.Models;

namespace YoloSharpOnnx.Inference.Detect
{
    public interface IDetPostprocess
    {
        public List<DetectionResult> PostProcess(OrtValue outputValue, PreDetectResult preResult, YoloConfig yoloConfig);
    }
}
