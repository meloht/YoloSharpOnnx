using Microsoft.ML.OnnxRuntime;
using System;
using System.Collections.Generic;
using System.Text;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Models;

namespace YoloSharpOnnx.Inference.Detect
{
    public interface IDetPostprocess
    {
        public List<DetectionResult> PostProcess(OrtValue outputValue, PreResult preResult, YoloConfig yoloConfig);
    }
}
